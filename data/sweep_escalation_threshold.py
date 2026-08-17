"""
Sweep candidate values for escalation_judge.py's
ESCALATION_SIMILARITY_THRESHOLD against data/eval_tune.csv, and report
accuracy / false-escalation-rate / false-confidence-rate at each one.

Why this exists: escalation_judge.py currently ships with
ESCALATION_SIMILARITY_THRESHOLD = 0.75 as an explicitly PROVISIONAL
default. This script is what turns that into a real, data-justified value
-- same principle as the classifier's rubric and the retriever's 0.55
threshold: "justify with data, not intuition" (README).

What it evaluates: for each of the 50 hand-labeled tickets in
eval_tune.csv, run the REAL Classifier and Retriever (not the hand labels
-- the actual, possibly-imperfect pipeline output), then test each
candidate threshold's escalation decision against the ticket's
hand-labeled expected_escalate. Classifier and Retriever are each called
ONCE per ticket and cached, since their output doesn't depend on the
threshold being tested -- only the Judge's rule does. The Resolver is
deliberately NOT called here: resolution_status is fully determined by
whether the Retriever found a match, so calling the Resolver would only
spend Groq calls without adding information the Judge's rule uses.

Metrics per threshold candidate:
  - Accuracy: predicted escalate decision matches expected_escalate.
  - False-escalation rate: predicted=Yes but expected=No, as a fraction of
    all expected=No tickets. This is the "annoying but cheap" error --
    a ticket that could've been auto-resolved goes to a human instead.
  - False-confidence rate: predicted=No but expected=Yes, as a fraction of
    all expected=Yes tickets. This is the COSTLY error per the project's
    core principle (README: "false escalation is cheaper than false
    confidence") -- a ticket that should have gone to a human gets
    auto-resolved instead. This rate should weigh most heavily in picking
    a threshold, even at the cost of some accuracy or a higher
    false-escalation rate.

Usage:
    python data/sweep_escalation_threshold.py
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import SessionLocal
from app.agents.classifier import classify_ticket
from app.agents.retriever import retrieve_resolution
from app.agents.escalation_judge import ALWAYS_ESCALATE_PRIORITIES

INPUT_PATH = "data/eval_tune.csv"
RESULTS_PATH = "data/escalation_threshold_sweep_results.csv"

# Candidate thresholds to test. Range chosen to bracket the retriever's own
# base threshold (0.55) up near-certain territory (0.95); every real
# similarity score observed in Week 2/3 evals so far has fallen in this band.
CANDIDATE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def predict_escalate(resolution_status: str, similarity: float | None, priority: str, threshold: float) -> bool:
    """Same rule logic as escalation_judge.judge_escalation, but taking the
    threshold as a parameter so it can be swept."""
    if resolution_status != "resolved" or similarity is None:
        return True
    if similarity < threshold:
        return True
    if priority in ALWAYS_ESCALATE_PRIORITIES:
        return True
    return False


def run_sweep():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing_labels = [r["subject"] for r in rows if not r.get("expected_escalate", "").strip()]
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} rows in {INPUT_PATH} have no expected_escalate "
            f"label -- label every row before running the sweep."
        )

    print(f"Running Classifier + Retriever once per ticket ({len(rows)} tickets)...\n")

    ticket_data = []
    for i, row in enumerate(rows, 1):
        classification = classify_ticket(row["subject"], row["description"])
        ticket_text = f"{row['subject']}. {row['description']}"

        # A fresh session per ticket, not one held open for the whole loop.
        # Neon's pooled connections can drop during the idle gap while
        # classify_ticket() is waiting on the Groq API -- pool_pre_ping
        # only validates a connection when it's freshly checked out, so a
        # single long-lived session can't recover from a mid-session drop.
        # One retry on top of that in case of a genuinely transient error.
        match = None
        last_error = None
        for attempt in range(2):
            db = SessionLocal()
            try:
                match = retrieve_resolution(db, ticket_text)
                last_error = None
                break
            except Exception as e:
                last_error = e
            finally:
                db.close()
        if last_error is not None:
            print(f"[{i}/{len(rows)}] \"{row['subject']}\" -> RETRIEVAL FAILED after retry: {last_error}")
            continue

        resolution_status = "resolved" if match is not None else "no_match"
        similarity = match["similarity"] if match else None

        expected_escalate = row["expected_escalate"].strip().lower() == "yes"

        ticket_data.append({
            "subject": row["subject"],
            "predicted_priority": classification["priority"],
            "resolution_status": resolution_status,
            "similarity": similarity,
            "expected_escalate": expected_escalate,
        })
        print(f"[{i}/{len(rows)}] \"{row['subject']}\" -> "
              f"priority={classification['priority']}, "
              f"similarity={similarity}, expected_escalate={expected_escalate}")

    print("\n" + "=" * 78)
    if len(ticket_data) < len(rows):
        print(f"NOTE: {len(rows) - len(ticket_data)} ticket(s) failed retrieval even after retry "
              f"and were excluded from the sweep below (see FAILED lines above).\n")
    print(f"{'Threshold':>10} | {'Accuracy':>9} | {'False-Escalation Rate':>22} | {'False-Confidence Rate':>22}")
    print("-" * 78)

    sweep_results = []
    n_expected_no = sum(1 for t in ticket_data if not t["expected_escalate"])
    n_expected_yes = sum(1 for t in ticket_data if t["expected_escalate"])

    for threshold in CANDIDATE_THRESHOLDS:
        correct = 0
        false_escalations = 0   # predicted Yes, expected No
        false_confidences = 0   # predicted No, expected Yes

        for t in ticket_data:
            predicted = predict_escalate(t["resolution_status"], t["similarity"], t["predicted_priority"], threshold)
            expected = t["expected_escalate"]

            if predicted == expected:
                correct += 1
            elif predicted and not expected:
                false_escalations += 1
            elif not predicted and expected:
                false_confidences += 1

        accuracy = correct / len(ticket_data)
        fer = false_escalations / n_expected_no if n_expected_no else 0.0
        fcr = false_confidences / n_expected_yes if n_expected_yes else 0.0

        sweep_results.append({
            "threshold": threshold, "accuracy": accuracy,
            "false_escalation_rate": fer, "false_confidence_rate": fcr,
            "false_escalations": false_escalations, "false_confidences": false_confidences,
        })
        print(f"{threshold:>10.2f} | {accuracy:>8.1%} | {fer:>21.1%} | {fcr:>21.1%}")

    print("=" * 78)

    # Recommend the threshold with the lowest false-confidence rate (the
    # costly error), breaking ties by highest accuracy.
    best = min(sweep_results, key=lambda r: (r["false_confidence_rate"], -r["accuracy"]))
    print(f"\nLowest false-confidence rate: threshold={best['threshold']} "
          f"(false_confidence_rate={best['false_confidence_rate']:.1%}, "
          f"accuracy={best['accuracy']:.1%})")
    print(
        "\nThis is a starting recommendation, not an automatic choice -- "
        "review the full table above. If multiple thresholds tie on "
        "false-confidence rate, prefer the lower false-escalation rate "
        "(fewer tickets needlessly sent to a human)."
    )

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
        writer.writeheader()
        writer.writerows(sweep_results)
    print(f"\nFull sweep table saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_sweep()