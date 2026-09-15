import os
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db import models

logger = logging.getLogger(__name__)

# We no longer cache models globally because 512MB is too small to hold 
# FastAPI + PyTorch + SentenceTransformer + CrossEncoder simultaneously.
# Instead, we will load them sequentially and explicitly garbage collect them.
import gc

# Below this cosine similarity, we don't trust the match. This is a first
# guess — Week 3's threshold sweep (for the Escalation Judge) will tell us
# if this needs to move.
SIMILARITY_THRESHOLD = 0.55
TOP_K_CANDIDATES = 5


def retrieve_resolution(db: Session, ticket_text: str) -> dict | None:
    """
    Embeds the incoming ticket text and searches the knowledge base for the
    closest match using pgvector's cosine distance operator.
    
    Then, re-ranks the top K candidates using a Cross-Encoder to fix
    the lexical-overlap limitations of the base embedding model.

    Returns a dict with the matched resolution and its similarity score,
    or None if nothing scores above SIMILARITY_THRESHOLD — a deliberate
    "I don't know" result rather than forcing a weak match.
    """
    # Import locally to avoid massive memory spikes during Uvicorn startup
    import torch
    torch.set_num_threads(1)
    from sentence_transformers import SentenceTransformer, CrossEncoder

    logger.info("Loading SentenceTransformer into memory...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    query_embedding = embed_model.encode(ticket_text).tolist()
    
    # 1. FETCH TOP K CANDIDATES
    # We query the DB here so we can delete the embed_model from memory ASAP
    distance_col = models.Resolution.embedding.cosine_distance(query_embedding)
    results = (
        db.query(models.Resolution, distance_col.label("distance"))
        .order_by(distance_col)
        .limit(TOP_K_CANDIDATES)
        .all()
    )

    # Free up 100MB+ of RAM before loading the CrossEncoder!
    del embed_model
    gc.collect()

    if not results:
        logger.warning("Knowledge base is empty — did you run seed_knowledge_base.py?")
        return None

    # 2. CROSS-ENCODER RERANKING
    logger.info("Loading CrossEncoder into memory...")
    cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
    
    # Pair the incoming ticket with each candidate's issue summary
    pairs = [[ticket_text, res.Resolution.issue_summary] for res in results]
    
    # Predict semantic similarity scores (0 to 1 for stsb models)
    cross_scores = cross_enc.predict(pairs)
    
    # Free up 100MB+ of RAM
    del cross_enc
    gc.collect()
    
    # Find the candidate with the highest cross-encoder score
    best_idx = cross_scores.argmax()
    best_resolution = results[best_idx].Resolution
    best_score = float(cross_scores[best_idx])
    
    logger.info(
        f"Reranker elevated '{best_resolution.issue_summary}' "
        f"(pgvector distance: {results[best_idx].distance:.3f}, "
        f"cross-score: {best_score:.3f})"
    )

    # 3. THRESHOLD GATING
    if best_score < SIMILARITY_THRESHOLD:
        logger.info(f"Best match cross-score {best_score:.2f} below threshold, no confident match.")
        return None

    return {
        "matched_issue": best_resolution.issue_summary,
        "resolution_text": best_resolution.resolution_text,
        "category": best_resolution.category,
        "similarity": round(best_score, 3),
    }
