"""
Diagnostic: print the RAW best-match similarity for every ticket in
eval_tune.csv, bypassing retriever.py's SIMILARITY_THRESHOLD cutoff.

retrieve_resolution() returns None whenever the best match scores below
0.55, which hides the actual number -- useful in production, unhelpful
when debugging why every ticket is coming back with no match. This script
does the same embedding + cosine-distance query directly, but always
prints the top match and its real similarity score, whatever it is.

Usage:
    python data/diagnose_retrieval_scores.py
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import SessionLocal
from app.db import models
from app.agents.retriever import _embedding_model

INPUT_PATH = "data/eval_tune.csv"


def diagnose():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    db = SessionLocal()
    try:
        for i, row in enumerate(rows, 1):
            ticket_text = f"{row['subject']}. {row['description']}"
            query_embedding = _embedding_model.encode(ticket_text).tolist()

            distance_col = models.Resolution.embedding.cosine_distance(query_embedding)
            result = (
                db.query(models.Resolution, distance_col.label("distance"))
                .order_by(distance_col)
                .first()
            )

            if result is None:
                print(f"[{i}/{len(rows)}] KNOWLEDGE BASE EMPTY -- no rows returned at all")
                continue

            resolution, distance = result
            similarity = round(1 - distance, 3)
            print(f"[{i}/{len(rows)}] \"{row['subject']}\" -> "
                  f"best_similarity={similarity} (vs 0.55 threshold) -- "
                  f"closest KB entry: \"{resolution.issue_summary}\"")
    finally:
        db.close()


if __name__ == "__main__":
    diagnose()