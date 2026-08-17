"""
Pull a fresh batch of real tickets from the raw Kaggle
dataset for a larger, more statistically stable eval set.

The original 20 tickets in sample_tickets_labeled.csv were hand-written
 -- useful for getting the Classifier working quickly, but too
small to responsibly sweep an escalation threshold on . This script instead
samples real tickets from the actual dataset, so the expanded eval set
reflects real ticket phrasing and distribution, not synthetic examples.

Splits the sample into two files:
  - eval_tune.csv       (50 tickets) -- used to sweep/tune the Escalation
                          Judge's threshold
  - eval_holdout.csv     (20 tickets) -- never touched during tuning; used
                          only to report final numbers, so the reported
                          accuracy isn't inflated by having been part of the
                          same search that picked the threshold

Both files come out with empty expected_category / expected_priority /
expected_escalate columns -- fill these in by hand, the same way
sample_tickets_labeled.csv was labeled. Filling in expected_escalate is a
judgment call: would you trust the pipeline to auto-resolve this ticket
from a KB match alone, or does it need a human regardless of what the
Retriever finds (e.g. genuinely ambiguous wording, high-stakes category)?

IMPORTANT -- this writes CSVs via the csv module with proper quoting
(QUOTE_MINIMAL), not manual string concatenation. This is deliberate: the
CSV-quoting bug (an unescaped comma in a description shifted every
column after it) happened because commas inside free-text fields weren't
quoted. Always open the output files afterward and spot-check a few rows
before labeling, in case a description contains a comma or quote character
pandas/csv didn't handle the way you expect.

Requires the raw Kaggle CSV already downloaded to data/raw/tickets.csv
(see data/clean_data.py's docstring for the download link). Not committed
to the repo -- data/raw/ is gitignored.

Usage:
    python data/pull_new_tickets.py
"""
import os
import random

import pandas as pd

RAW_PATH = "data/raw/tickets.csv"
EXISTING_LABELED_PATH = "data/sample_tickets_labeled.csv"
TUNE_OUTPUT_PATH = "data/eval_tune.csv"
HOLDOUT_OUTPUT_PATH = "data/eval_holdout.csv"

TUNE_SIZE = 50
HOLDOUT_SIZE = 20
RANDOM_SEED = 42  # fixed seed so this sample is reproducible, not re-randomized every run

OUTPUT_COLUMNS = [
    "subject", "description", "product", "channel",
    "expected_category", "expected_priority", "expected_escalate",
]


def pull_new_tickets():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Download the Kaggle Customer Support Ticket "
            f"Dataset (see data/clean_data.py docstring) and place it there first."
        )

    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} raw tickets from {RAW_PATH}")

    # Same column rename as clean_data.py, kept consistent so both scripts
    # agree on what "subject"/"description"/etc. mean.
    df = df.rename(columns={
        "Ticket Subject": "subject",
        "Ticket Description": "description",
        "Product Purchased": "product",
        "Ticket Channel": "channel",
    })

    keep_cols = ["subject", "description", "product", "channel"]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected columns {missing} not found in raw CSV after renaming. "
            f"Available columns: {list(df.columns)}"
        )
    df = df[keep_cols].dropna(subset=["subject", "description"])

    # Drop exact-duplicate tickets within the raw dataset itself (some
    # Kaggle support datasets have repeated template tickets).
    before = len(df)
    df = df.drop_duplicates(subset=["subject", "description"])
    print(f"Dropped {before - len(df)} duplicate rows within the raw dataset.")

    # Exclude anything whose subject overlaps the existing 20 hand-written
    # tickets, so the expanded eval set doesn't double up on the same cases.
    if os.path.exists(EXISTING_LABELED_PATH):
        existing = pd.read_csv(EXISTING_LABELED_PATH)
        existing_subjects = set(existing["subject"].str.strip().str.lower())
        before = len(df)
        df = df[~df["subject"].str.strip().str.lower().isin(existing_subjects)]
        print(f"Excluded {before - len(df)} rows overlapping the existing 20 labeled tickets.")

    total_needed = TUNE_SIZE + HOLDOUT_SIZE
    if len(df) < total_needed:
        raise ValueError(
            f"Only {len(df)} eligible tickets available after filtering, "
            f"need {total_needed}. Check the raw CSV or loosen the dedup filter."
        )

    sample = df.sample(n=total_needed, random_state=RANDOM_SEED).reset_index(drop=True)

    # Add empty label columns for manual labeling.
    for col in ("expected_category", "expected_priority", "expected_escalate"):
        sample[col] = ""

    tune_set = sample.iloc[:TUNE_SIZE]
    holdout_set = sample.iloc[TUNE_SIZE:]

    tune_set.to_csv(TUNE_OUTPUT_PATH, index=False, columns=OUTPUT_COLUMNS)
    holdout_set.to_csv(HOLDOUT_OUTPUT_PATH, index=False, columns=OUTPUT_COLUMNS)

    print(f"\nWrote {len(tune_set)} tickets to {TUNE_OUTPUT_PATH} (for threshold tuning)")
    print(f"Wrote {len(holdout_set)} tickets to {HOLDOUT_OUTPUT_PATH} (held out for final reporting)")
    print(
        "\nNext step: open both CSVs and fill in expected_category, "
        "expected_priority, and expected_escalate by hand -- same process "
        "as sample_tickets_labeled.csv, plus the new escalate column."
    )


if __name__ == "__main__":
    pull_new_tickets()