"""
Quick diagnostic: for a few tickets whose subject looks like noise (mismatched
to the real complaint), compare retrieval similarity with subject+description
vs. description-only. Run from the project root:

    python data/diagnose_subject_noise.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import SessionLocal
from app.agents.retriever import retrieve_resolution

# A few clear cases from the tune-set gap analysis where the subject looks
# like noise relative to the actual complaint.
CASES = [
    {
        "subject": "Installation support",
        "description": (
            "I've noticed a sudden decrease in battery life on my Roomba "
            "Robot Vacuum. It used to last much longer."
        ),
    },
    {
        "subject": "Installation support",
        "description": (
            "I've recently set up my GoPro Hero, but it fails to connect to "
            "any available networks. What steps should I take to "
            "troubleshoot this issue? I need assistance as soon as possible "
            "because it's affecting my work and productivity."
        ),
    },
    {
        "subject": "Delivery problem",
        "description": (
            "This problem started occurring after the recent software "
            "update. I haven't made any other changes to the device."
        ),
    },
]

db = SessionLocal()
try:
    for case in CASES:
        subject, description = case["subject"], case["description"]

        with_subject = retrieve_resolution(db, f"{subject}. {description}")
        without_subject = retrieve_resolution(db, description)

        print(f"Subject: {subject!r}")
        print(f"Description: {description[:80]!r}...")
        print(
            f"  WITH subject prefix:    "
            f"{'similarity=' + str(with_subject['similarity']) if with_subject else 'NO MATCH'}"
        )
        print(
            f"  WITHOUT subject prefix: "
            f"{'similarity=' + str(without_subject['similarity']) if without_subject else 'NO MATCH'}"
        )
        print()
finally:
    db.close()