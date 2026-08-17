"""
Embeds each knowledge base entry using sentence-transformers and
inserts it into the `resolutions` table. Run this once to populate the
knowledge base; safe to re-run (it clears and re-seeds each time).

Usage:
    python data/seed_knowledge_base.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer

from app.db.database import SessionLocal, Base, engine
from app.db import models
from data.seed_knowledge_base_data import KNOWLEDGE_BASE_SEED

print("Loading embedding model (all-MiniLM-L6-v2)... this downloads ~90MB on first run.")
model = SentenceTransformer("all-MiniLM-L6-v2")


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Clear existing entries so this script is safely re-runnable
    deleted = db.query(models.Resolution).delete()
    print(f"Cleared {deleted} existing knowledge base entries.")

    for entry in KNOWLEDGE_BASE_SEED:
        embedding = model.encode(entry["issue_summary"]).tolist()
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
