"""
Final evaluation against data/eval_holdout.csv -- the 20
tickets held out from threshold tuning.

Unlike data/sweep_escalation_threshold.py (which skipped the Resolver and
tested many threshold candidates against the TUNE set), this script runs
the REAL, locked-in production pipeline -- app.graph.run_triage_pipeline(),
the exact function main.py's /submit-ticket endpoint calls -- against every
holdout ticket, using the actual ESCALATION_SIMILARITY_THRESHOLD (0.60)
now baked into escalation_judge.py. This is deliberately the strongest,
most production-faithful validation available: no shortcuts, no threshold
parameter, just the real system end to end.

These 20 tickets were never used to choose the threshold, so the numbers
here are the honest, reportable "how well does this actually work"
result -- not numbers that were implicitly optimized for by the tuning
process (which is exactly why the tune/holdout split exists).

Reports, against the hand-labeled ground truth in eval_holdout.csv:
  - Category accuracy (Classifier)
  - Priority accuracy (Classifier)
  - Escalation accuracy, false-escalation rate, false-confidence rate
    (Escalation Judge, end to end)

Usage:
    python data/final_holdout_eval.py
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import SessionLocal
from app.graph import run_triage_pipeline

INPUT_PATH = "data/eval_holdout.csv"
RESULTS_PATH = "data/final_holdout_eval_results.csv"


def run_final_eval():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing_labels = [
        r["subject"] for r in rows
        if not r.get("expected_escalate", "").strip()
        or not r.get("expected_category", "").strip()
        or not r.get("expected_priority", "").strip()
    ]
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} rows in {INPUT_PATH} are missing a label -- "
            f"label every row before running the final eval."
        )

    print(f"Running the full pipeline (Classifier -> Retriever -> Resolver -> "
          f"Escalation Judge) against {len(rows)} held-out tickets...\n")

    results = []
    category_correct = 0
    priority_correct = 0
    escalate_correct = 0
    false_escalations = 0
    false_confidences = 0
    n_expected_no = 0
    n_expected_yes = 0

    for i, row in enumerate(rows, 1):
        # Fresh session per ticket -- avoids the Neon idle-connection drop
        # hit during the threshold sweep. Even with a fresh session, a
        # transient drop can still happen (e.g. Neon's compute scaling
        # down/up between slower per-ticket work now that the Resolver's
        # extra Groq call is in the loop too) -- retry the whole per-ticket
        # pipeline call once before giving up on that ticket, and keep
        # going rather than crashing the entire run.
        pipeline_result = None
        last_error = None
        for attempt in range(2):
            db = SessionLocal()
            try:
                pipeline_result = run_triage_pipeline(db, row["subject"], row["description"])
                last_error = None
                break
            except Exception as e:
                last_error = e
            finally:
                db.close()

        if last_error is not None:
            print(f"[{i}/{len(rows)}] \"{row['subject']}\" -> FAILED after retry: {last_error}")
            continue

        expected_category = row["expected_category"].strip()
        expected_priority = row["expected_priority"].strip()
        expected_escalate = row["expected_escalate"].strip().lower() == "yes"

        predicted_category = pipeline_result["category"]
        predicted_priority = pipeline_result["priority"]
        predicted_escalate = pipeline_result["escalate"]

        cat_match = predicted_category == expected_category
        pri_match = predicted_priority == expected_priority
        esc_match = predicted_escalate == expected_escalate

        category_correct += cat_match
        priority_correct += pri_match
        escalate_correct += esc_match
        if expected_escalate:
            n_expected_yes += 1
        else:
            n_expected_no += 1
        if predicted_escalate and not expected_escalate:
            false_escalations += 1
        elif not predicted_escalate and expected_escalate:
            false_confidences += 1

        print(f"[{i}/{len(rows)}] \"{row['subject']}\" -> "
              f"category={predicted_category} ({'OK' if cat_match else 'MISS, expected ' + expected_category}), "
              f"priority={predicted_priority} ({'OK' if pri_match else 'MISS, expected ' + expected_priority}), "
              f"escalate={predicted_escalate} ({'OK' if esc_match else 'MISS, expected ' + str(expected_escalate)})")

        results.append({
            "subject": row["subject"],
            "predicted_category": predicted_category, "expected_category": expected_category,
            "predicted_priority": predicted_priority, "expected_priority": expected_priority,
            "predicted_escalate": predicted_escalate, "expected_escalate": expected_escalate,
            "similarity": pipeline_result["retrieved_match"]["similarity"] if pipeline_result["retrieved_match"] else None,
            "escalation_reason": pipeline_result["escalation_reason"],
        })

    n = len(results)
    if n < len(rows):
        print(f"\nNOTE: {len(rows) - n} ticket(s) failed even after retry and were excluded "
              f"from the metrics below.")
    print("\n" + "=" * 60)
    print(f"Category accuracy:            {category_correct}/{n}  ({category_correct/n:.1%})")
    print(f"Priority accuracy:             {priority_correct}/{n}  ({priority_correct/n:.1%})")
    print(f"Escalation decision accuracy:  {escalate_correct}/{n}  ({escalate_correct/n:.1%})")
    if n_expected_no:
        print(f"False-escalation rate:         {false_escalations}/{n_expected_no}  ({false_escalations/n_expected_no:.1%})")
    if n_expected_yes:
        print(f"False-confidence rate:         {false_confidences}/{n_expected_yes}  ({false_confidences/n_expected_yes:.1%})")
    print("=" * 60)

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_final_eval()