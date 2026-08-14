"""
Week 1, Day 5: Run the Classifier Agent against a labeled batch of tickets
and measure accuracy. This calls classify_ticket() directly — no FastAPI
server or database needed, so you can run this standalone.

Usage:
    python data/test_classifier.py
"""
import sys
import os
import csv

# Allow running this script directly from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.classifier import classify_ticket

INPUT_PATH = "data/sample_tickets_labeled.csv"
RESULTS_PATH = "data/classifier_eval_results.csv"


def run_eval():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Running classifier against {len(rows)} labeled tickets...\n")

    results = []
    category_correct = 0
    priority_correct = 0
    both_correct = 0

    for i, row in enumerate(rows, 1):
        predicted = classify_ticket(row["subject"], row["description"])

        cat_match = predicted["category"] == row["expected_category"]
        pri_match = predicted["priority"] == row["expected_priority"]

        if cat_match:
            category_correct += 1
        if pri_match:
            priority_correct += 1
        if cat_match and pri_match:
            both_correct += 1

        status = "✅" if cat_match and pri_match else ("⚠️" if cat_match or pri_match else "❌")
        print(f"{status} [{i}/{len(rows)}] \"{row['subject']}\"")
        print(f"    Expected: {row['expected_category']} / {row['expected_priority']}")
        print(f"    Predicted: {predicted['category']} / {predicted['priority']}\n")

        results.append({
            "subject": row["subject"],
            "expected_category": row["expected_category"],
            "predicted_category": predicted["category"],
            "category_match": cat_match,
            "expected_priority": row["expected_priority"],
            "predicted_priority": predicted["priority"],
            "priority_match": pri_match,
        })

    n = len(rows)
    print("=" * 50)
    print(f"Category accuracy:      {category_correct}/{n}  ({category_correct/n*100:.1f}%)")
    print(f"Priority accuracy:      {priority_correct}/{n}  ({priority_correct/n*100:.1f}%)")
    print(f"Both correct:           {both_correct}/{n}  ({both_correct/n*100:.1f}%)")
    print("=" * 50)

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()
