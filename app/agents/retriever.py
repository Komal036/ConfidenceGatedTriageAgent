import gc
import json
import asyncio
import hashlib
import logging
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort
from sqlalchemy.orm import Session
import redis

from app.db import models
from app.config import settings

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.55
TOP_K_CANDIDATES = 5

_retriever_lock = asyncio.Semaphore(1)

def _get_redis():
    """Returns a Redis client or None if not configured."""
    if not settings.redis_url:
        return None
    return redis.from_url(settings.redis_url)

def _cache_key(ticket_text: str) -> str:
    return f"retriever:{hashlib.sha256(ticket_text.encode()).hexdigest()}"

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, -1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def _l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)

def _sync_retrieve_resolution(db: Session, ticket_text: str) -> dict | None:
    # --- Check Cache First ---
    redis_client = _get_redis()
    cache_key = _cache_key(ticket_text)
    if redis_client:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info("Retriever cache hit.")
                return json.loads(cached_result)
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")

    # --- Embedding Generation ---
    logger.info("Loading Embedding Tokenizer and Model into memory...")
    embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
    
    embed_tokenizer_path = hf_hub_download(repo_id=embed_model_id, filename="tokenizer.json")
    embed_onnx_path = hf_hub_download(repo_id=embed_model_id, filename="onnx/model.onnx")
    
    embed_tokenizer = Tokenizer.from_file(embed_tokenizer_path)
    embed_tokenizer.enable_truncation(max_length=256)
    embed_tokenizer.enable_padding()
    
    encoded = embed_tokenizer.encode(ticket_text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoded.type_ids], dtype=np.int64)
    
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    
    embed_session = ort.InferenceSession(embed_onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }
    ort_outs = embed_session.run(None, ort_inputs)
    token_embeddings = ort_outs[0] 
    
    sentence_embeddings = _mean_pooling(token_embeddings, attention_mask)
    sentence_embeddings = _l2_normalize(sentence_embeddings)
    query_embedding = sentence_embeddings[0].tolist()
    
    del embed_session
    del embed_tokenizer
    gc.collect()

    # --- Fetch Candidates from DB ---
    distance_col = models.Resolution.embedding.cosine_distance(query_embedding)
    results = (
        db.query(models.Resolution, distance_col.label("distance"))
        .order_by(distance_col)
        .limit(TOP_K_CANDIDATES)
        .all()
    )

    if not results:
        logger.warning("Knowledge base is empty — did you run seed_knowledge_base.py?")
        return None

    # --- Cross-Encoder Reranking ---
    logger.info("Loading CrossEncoder into memory...")
    cross_model_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    
    cross_tokenizer_path = hf_hub_download(repo_id=cross_model_id, filename="tokenizer.json")
    cross_onnx_path = hf_hub_download(repo_id=cross_model_id, filename="onnx/model.onnx")
    
    cross_tokenizer = Tokenizer.from_file(cross_tokenizer_path)
    cross_tokenizer.enable_truncation(max_length=512)
    cross_tokenizer.enable_padding()
    
    pairs = [[ticket_text, res.Resolution.issue_summary] for res in results]
    
    encoded_pairs = cross_tokenizer.encode_batch(pairs)
    
    cross_input_ids = np.array([enc.ids for enc in encoded_pairs], dtype=np.int64)
    cross_attention_mask = np.array([enc.attention_mask for enc in encoded_pairs], dtype=np.int64)
    cross_token_type_ids = np.array([enc.type_ids for enc in encoded_pairs], dtype=np.int64)
    
    cross_session = ort.InferenceSession(cross_onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"])
    
    cross_ort_inputs = {
        "input_ids": cross_input_ids,
        "attention_mask": cross_attention_mask,
        "token_type_ids": cross_token_type_ids
    }
    
    cross_outs = cross_session.run(None, cross_ort_inputs)
    cross_scores = cross_outs[0].flatten()
    
    del cross_session
    del cross_tokenizer
    gc.collect()
    
    best_idx = cross_scores.argmax()
    best_resolution = results[best_idx].Resolution
    best_score = float(cross_scores[best_idx])
    
    logger.info(
        f"Reranker elevated '{best_resolution.issue_summary}' "
        f"(pgvector distance: {results[best_idx].distance:.3f}, "
        f"cross-score: {best_score:.3f})"
    )

    if best_score < SIMILARITY_THRESHOLD:
        logger.info(f"Best match cross-score {best_score:.2f} below threshold, no confident match.")
        result = None
    else:
        result = {
            "matched_issue": best_resolution.issue_summary,
            "resolution_text": best_resolution.resolution_text,
            "category": best_resolution.category,
            "similarity": round(best_score, 3),
        }

    # --- Cache Result ---
    if redis_client:
        try:
            redis_client.setex(cache_key, 3600, json.dumps(result))
        except Exception as e:
            logger.warning(f"Redis cache set error: {e}")

    return result

async def retrieve_resolution(db: Session, ticket_text: str) -> dict | None:
    """
    Embeds the incoming ticket text and searches the knowledge base for the
    closest match using pgvector's cosine distance operator.
    
    Then, re-ranks the top K candidates using a Cross-Encoder to fix
    the lexical-overlap limitations of the base embedding model.

    Returns a dict with the matched resolution and its similarity score,
    or None if nothing scores above SIMILARITY_THRESHOLD — a deliberate
    "I don't know" result rather than forcing a weak match.

    WARNING: The embeddings MUST remain 384-dimensional and compatible with 
    the existing pgvector data seeded by seed_knowledge_base.py using all-MiniLM-L6-v2. 
    If the ONNX model produces different embeddings than the PyTorch model for the same input, 
    the user will need to re-run seed_knowledge_base.py.
    """
    async with _retriever_lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_retrieve_resolution, db, ticket_text)
