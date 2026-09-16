"""
Embeds each knowledge base entry using ONNX runtime and
inserts it into the `resolutions` table. Run this once to populate the
knowledge base; safe to re-run (it clears and re-seeds each time).

Usage:
    python data/seed_knowledge_base.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from app.db.database import SessionLocal, Base, engine
from app.db import models
from data.seed_knowledge_base_data import KNOWLEDGE_BASE_SEED

print("Loading embedding model (all-MiniLM-L6-v2) via ONNX... this downloads ~90MB on first run.")

embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer_path = hf_hub_download(repo_id=embed_model_id, filename="tokenizer.json")
embed_onnx_path = hf_hub_download(repo_id=embed_model_id, filename="onnx/model.onnx")

tokenizer = Tokenizer.from_file(embed_tokenizer_path)
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding()

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session = ort.InferenceSession(embed_onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"])


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


def encode(text: str):
    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoded.type_ids], dtype=np.int64)
    
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids
    }
    ort_outs = session.run(None, ort_inputs)
    token_embeddings = ort_outs[0] 
    
    sentence_embeddings = _mean_pooling(token_embeddings, attention_mask)
    sentence_embeddings = _l2_normalize(sentence_embeddings)
    return sentence_embeddings[0].tolist()


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Clear existing entries so this script is safely re-runnable
    deleted = db.query(models.Resolution).delete()
    print(f"Cleared {deleted} existing knowledge base entries.")

    for entry in KNOWLEDGE_BASE_SEED:
        embedding = encode(entry["issue_summary"])
        resolution = models.Resolution(
            category=entry["category"],
            issue_summary=entry["issue_summary"],
            resolution_text=entry["resolution_text"],
            embedding=embedding,
        )
        db.add(resolution)

    db.commit()
    count = db.query(models.Resolution).count()
    db.close()
    print(f"Seeded {count} knowledge base entries with embeddings.")


if __name__ == "__main__":
    seed()
