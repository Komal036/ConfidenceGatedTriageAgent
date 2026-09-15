import os
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db import models

logger = logging.getLogger(__name__)

import torch
torch.set_num_threads(1)

# Loaded once at import time, reused across requests — same pattern as the
# Groq client in classifier.py, for the same reason: expensive setup done
# once, not per-request.
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
# Add a cross-encoder for semantic reranking of the top K results.
# Using the stsb model because it naturally outputs scores between 0 and 1,
# matching our pipeline's expected similarity thresholding.
_cross_encoder = CrossEncoder("cross-encoder/stsb-MiniLM-L6-v2", device="cpu")

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
    query_embedding = _embedding_model.encode(ticket_text).tolist()

    # pgvector's <=> operator returns cosine DISTANCE (0 = identical, 2 = opposite).
    distance_col = models.Resolution.embedding.cosine_distance(query_embedding)

    # 1. FETCH TOP K CANDIDATES
    results = (
        db.query(models.Resolution, distance_col.label("distance"))
        .order_by(distance_col)
        .limit(TOP_K_CANDIDATES)
        .all()
    )

    if not results:
        logger.warning("Knowledge base is empty — did you run seed_knowledge_base.py?")
        return None

    # 2. CROSS-ENCODER RERANKING
    # Pair the incoming ticket with each candidate's issue summary
    pairs = [[ticket_text, res.Resolution.issue_summary] for res in results]
    
    # Predict semantic similarity scores (0 to 1 for stsb models)
    cross_scores = _cross_encoder.predict(pairs)
    
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
