"""
Targeted diagnostic for the row-3 scoring anomaly from eval_tune.csv:

  subject: "Product compatibility"
  description: "I'm unable to find the option to perform the desired
      action in the MacBook Pro. Could you please guide me through the
      steps?"

This scored only 0.236 against the KB entry "User can't locate a specific
feature, setting, or option and needs step-by-step navigation guidance" --
suspiciously low for two near-paraphrases. This script checks three things
directly, bypassing the DB entirely, to isolate the cause:

  1. Raw cosine similarity between the ticket DESCRIPTION ONLY and the KB
     entry's issue_summary text.
  2. Raw cosine similarity between SUBJECT+DESCRIPTION (what the pipeline
     actually embeds) and the KB entry's issue_summary text.
  3. Raw cosine similarity between the description and a few OTHER KB
     entries, to sanity-check the embedding model isn't just returning
     uniformly low scores for everything (which would point to a model/
     setup issue rather than a phrasing issue).

Usage:
    python data/diagnose_row3_anomaly.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from app.agents.retriever import _embedding_model

SUBJECT = "Product compatibility"
DESCRIPTION = (
    "I'm unable to find the option to perform the desired action in the "
    "MacBook Pro. Could you please guide me through the steps?"
)

KB_TARGET = (
    "User can't locate a specific feature, setting, or option and needs "
    "step-by-step navigation guidance"
)

# A few unrelated KB entries, as a sanity-check baseline.
KB_OTHERS = {
    "WiFi disconnecting": "WiFi keeps disconnecting intermittently",
    "Battery draining": "Battery draining much faster than usual",
    "Charged twice": "Charged twice for the same subscription period",
}


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    desc_only_emb = _embedding_model.encode(DESCRIPTION).tolist()
    subj_desc_emb = _embedding_model.encode(f"{SUBJECT}. {DESCRIPTION}").tolist()
    target_emb = _embedding_model.encode(KB_TARGET).tolist()

    print("Ticket description:", DESCRIPTION)
    print("Target KB entry:   ", KB_TARGET)
    print()
    print(f"1. description-only vs target KB entry:      {cosine_sim(desc_only_emb, target_emb):.3f}")
    print(f"2. subject+description vs target KB entry:    {cosine_sim(subj_desc_emb, target_emb):.3f}")
    print()
    print("3. Sanity check -- description-only vs unrelated KB entries (should score LOWER than #1 if the model is working correctly):")
    for name, text in KB_OTHERS.items():
        other_emb = _embedding_model.encode(text).tolist()
        print(f"   vs {name!r}: {cosine_sim(desc_only_emb, other_emb):.3f}")


if __name__ == "__main__":
    main()