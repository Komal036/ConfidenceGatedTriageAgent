import logging

from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db import models

logger = logging.getLogger(__name__)

# Loaded once at import time, reused across requests — same pattern as the
# Groq client in classifier.py, for the same reason: expensive setup done
# once, not per-request.
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Below this cosine similarity, we don't trust the match. This is a first
# guess — Week 3's threshold sweep (for the Escalation Judge) will tell us
# if this needs to move.
SIMILARITY_THRESHOLD = 0.55


def retrieve_resolution(db: Session, ticket_text: str) -> dict | None:
    """
    Embeds the incoming ticket text and searches the knowledge base for the
    closest match using pgvector's cosine distance operator.

    Returns a dict with the matched resolution and its similarity score,
    or None if nothing scores above SIMILARITY_THRESHOLD — a deliberate
    "I don't know" result rather than forcing a weak match. This is exactly
    the kind of signal the Escalation Judge will use in Week 3.
    """
    query_embedding = _embedding_model.encode(ticket_text).tolist()

    # pgvector's <=> operator returns cosine DISTANCE (0 = identical, 2 = opposite).
    # We convert to similarity (1 = identical, -1 = opposite) for readability,
    # since "similarity" is more intuitive to reason about than "distance".
    distance_col = models.Resolution.embedding.cosine_distance(query_embedding)

    result = (
        db.query(models.Resolution, distance_col.label("distance"))
        .order_by(distance_col)
        .first()
    )

    if result is None:
        logger.warning("Knowledge base is empty — did you run seed_knowledge_base.py?")
        return None

    resolution, distance = result
    similarity = 1 - distance

    if similarity < SIMILARITY_THRESHOLD:
        logger.info(f"Best match similarity {similarity:.2f} below threshold, no confident match.")
        return None

    return {
        "matched_issue": resolution.issue_summary,
        "resolution_text": resolution.resolution_text,
        "category": resolution.category,
        "similarity": round(similarity, 3),
    }
