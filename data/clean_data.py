"""
Day 1 script: clean and explore the Kaggle Customer Support Ticket Dataset.

Download the CSV manually from:
https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
and place it at data/raw/tickets.csv before running this.

Usage:
    python data/clean_data.py
"""
import pandas as pd

RAW_PATH = "data/raw/tickets.csv"
CLEAN_PATH = "data/clean_tickets.csv"


def clean_and_explore():
    df = pd.read_csv(RAW_PATH)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}\n")

    # Standardize column names to match our schema
    df = df.rename(columns={
        "Ticket Subject": "subject",
        "Ticket Description": "description",
        "Ticket Type": "category",
        "Ticket Priority": "priority",
        "Product Purchased": "product",
        "Ticket Channel": "channel",
        "Ticket Status": "status",
        "Resolution": "resolution",
    })

    keep_cols = ["subject", "description", "category", "priority", "product", "channel", "status", "resolution"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # This dataset's `resolution` field is sparse — flag how sparse before
    # deciding how much of your knowledge base needs to come from
    # hand-written supplementary tickets instead.
    resolution_fill_rate = df["resolution"].notna().mean() * 100
    print(f"'resolution' field is populated for {resolution_fill_rate:.1f}% of rows.\n")

    print("Category distribution:")
    print(df["category"].value_counts(), "\n")

    print("Priority distribution:")
    print(df["priority"].value_counts(), "\n")

    # Drop rows with missing subject/description — unusable for classification
    before = len(df)
    df = df.dropna(subset=["subject", "description"])
    print(f"Dropped {before - len(df)} rows with missing subject/description.")

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned data to {CLEAN_PATH} ({len(df)} rows).")


if __name__ == "__main__":
    clean_and_explore()
