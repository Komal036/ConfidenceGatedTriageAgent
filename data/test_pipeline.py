"""
Run the Retriever + Resolver against the same 20 labeled
tickets used for the classifier eval, and measure how they behave
end-to-end.

This is a different kind of eval than test_classifier.py: there's no
ground-truth "correct resolution text" to score a draft against, so instead
this measures the two things that actually matter for a confidence-gated
system:

1. Retrieval hit rate -- for how many tickets did the Retriever find a
   match above SIMILARITY_THRESHOLD at all?
2. Retrieval category agreement -- of the tickets that got a match, how
   often does the matched knowledge base entry's category agree with the
   ticket's actual (expected) category? A high hit rate with low category
   agreement would mean the threshold is too permissive -- it's finding
   *something*, but not the *right* something. This is exactly the number
   threshold sweep (for the Escalation Judge) will use as a
   starting point.

Requires the knowledge base to already be seeded (data/seed_knowledge_base.py)
and DATABASE_URL to point at a live Neon instance with pgvector enabled --
unlike test_classifier.py, this can't run fully offline.

Usage:
    python data/test_pipeline.py
"""
import sys
import os
import csv

# Allow running this script directly from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import SessionLocal
from app.agents.retriever import retrieve_resolution
from app.agents.resolver import resolve_ticket

INPUT_PATH = "data/sample_tickets_labeled.csv"
RESULTS_PATH = "data/pipeline_eval_results.csv"


def run_eval():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Running Retriever + Resolver against {len(rows)} labeled tickets...\n")

    db = SessionLocal()
    results = []
    hits = 0
    category_agrees = 0

    try:
        for i, row in enumerate(rows, 1):
            # Same subject+description concatenation the graph uses in
            # app/graph.py's retrieve_node -- keep the eval consistent with
            # how the pipeline actually calls the Retriever.
            ticket_text = f"{row['subject']}. {row['description']}"
            match = retrieve_resolution(db, ticket_text)

            cat_agrees = None
            if match is not None:
                hits += 1
                cat_agrees = match["category"] == row["expected_category"]
                if cat_agrees:
                    category_agrees += 1

            resolved = resolve_ticket(row["subject"], row["description"], match)

            status = "✅" if match and cat_agrees else ("⚠️" if match else "❌")
            print(f"{status} [{i}/{len(rows)}] \"{row['subject']}\"")
            if match:
                print(f"    Matched: {match['matched_issue']} "
                      f"(similarity={match['similarity']}, category={match['category']})")
                print(f"    Expected category: {row['expected_category']}")
            else:
                print("    No confident match found (below SIMILARITY_THRESHOLD).")
            print(f"    Resolver: status={resolved['status']}, tool_called={resolved['tool_called']}\n")

            results.append({
                "subject": row["subject"],
                "expected_category": row["expected_category"],
                "retrieval_hit": match is not None,
                "matched_issue": match["matched_issue"] if match else None,
                "match_similarity": match["similarity"] if match else None,
                "match_category": match["category"] if match else None,
                "category_agrees": cat_agrees,
                "resolver_status": resolved["status"],
                "tool_called": resolved["tool_called"],
                "draft_resolution": resolved["draft_resolution"],
            })
    finally:
        db.close()

    n = len(rows)
    print("=" * 50)
    print(f"Retrieval hit rate:            {hits}/{n}  ({hits / n * 100:.1f}%)")
    if hits:
        print(f"Category agreement (of hits):  {category_agrees}/{hits}  "
              f"({category_agrees / hits * 100:.1f}%)")
    print("=" * 50)

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()