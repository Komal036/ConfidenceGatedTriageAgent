"""
Clean the descriptions in eval_tune.csv / eval_holdout.csv.

Why this exists: the raw Kaggle "Customer Support Ticket Dataset" turns out
to have a known data-quality issue. Every description in the file is really
two things concatenated with no delimiter:

  1. An opening scenario sentence (e.g. "I'm having an issue with the
     {product_purchased}. Please assist.") followed, in many rows, by
     incoherent filler text -- unrelated sentences, code snippets, or
     marketing-sounding boilerplate that has nothing to do with the ticket.
  2. A closing "real" complaint sentence, drawn from a small recurring pool
     (~20 variants -- "I've noticed a sudden decrease in battery life on my
     {product_purchased}...", "I'm concerned about the security of...",
     etc.) that IS a coherent, plausible support complaint.

Both halves also contain literal, never-substituted `{product_purchased}`
and `{error_message}` placeholders.

This script:
  1. Searches each raw description for the LAST occurrence of any known
     closing-sentence template (the templates are hardcoded below, built by
     inspecting the actual pulled data -- see CLOSING_TEMPLATES).
  2. Keeps only the text from that point onward, discarding the garbled
     opening. This is a deliberate trade-off: it throws away some real
     signal in exchange for reliably discarding the incoherent filler,
     since the filler's position and content aren't predictable enough to
     strip out any other way.
  3. Substitutes {product_purchased} with the actual value from the
     `product` column, and {error_message} with a generic placeholder
     phrase (the real error text was never in the dataset to recover).
  4. Rows where no known template matches are left with cleaning_status =
     "no_template_match" and their FULL raw description kept, so nothing
     is silently thrown away -- these need a manual look before labeling.

Run this once per file, review the "cleaning_status" column, then label
expected_category / expected_priority / expected_escalate against the
`description` column (the cleaned one), not `raw_description`.

Usage:
    python data/clean_eval_descriptions.py data/eval_tune.csv
    python data/clean_eval_descriptions.py data/eval_holdout.csv
"""
import re
import sys

import pandas as pd

# Recurring closing-sentence templates found in the raw dataset. Order
# doesn't matter -- for each row we find every template that matches
# anywhere in the text and keep the one that starts LATEST (closest to the
# end), since garbled filler can itself coincidentally contain a
# template-like word without being the real complaint sentence.
CLOSING_TEMPLATES = [
    r"I've noticed a sudden decrease in battery life on my \{product_purchased\}\. It used to last much longer\.",
    r"This problem started occurring after the recent software update\. I haven't made any other changes to the device\.",
    r"I'm unable to find the option to perform the desired action in the \{product_purchased\}\. Could you please guide me through the steps\?",
    r"I've noticed that the issue occurs consistently when I use a specific feature or application on my \{product_purchased\}\.",
    r"I'm experiencing this issue on multiple devices of the same model, so it seems to be a widespread problem\.",
    r"I've already contacted customer support multiple times, but the issue remains unresolved\.",
    r"I've tried troubleshooting steps mentioned in the user manual, but the issue persists\.",
    r"I rely heavily on my \{product_purchased\} for my daily tasks, and this issue is hindering my productivity\.",
    r"I need assistance as soon as possible because it's affecting my work and productivity\.",
    r"I've recently updated the firmware of my \{product_purchased\}, and the issue started happening afterward\. Could it be related to the update\?",
    r"I've noticed a peculiar error message popping up on my \{product_purchased\} screen\. It says '\{error_message\}'\. What does it mean\?",
    r"I've tried using different cables, adapters, or peripherals with my \{product_purchased\}, but the issue persists\.",
    r"I'm concerned about the security of my \{product_purchased\} and would like to ensure that my data is safe\.",
    r"I've checked for any available software updates for my \{product_purchased\}, but there are none\.",
    r"I'm worried that the issue might be hardware-related and might require repair or replacement\.",
    r"I've performed a factory reset on my \{product_purchased\}, hoping it would resolve the problem, but it didn't help\.",
    r"I'm using the original charger that came with my \{product_purchased\}, but it's not charging properly\.",
    r"I've tried different settings and configurations on my \{product_purchased\}, but the issue persists\.",
    r"I'm not sure if this issue is specific to my device or if others have reported similar problems\.",
    r"I've followed the troubleshooting steps mentioned in the user manual, but the issue persists\.",
    r"The issue I'm facing is intermittent\. Sometimes it works fine, but other times it acts up unexpectedly\.",
    r"I've followed online tutorials and community forums to troubleshoot the issue, but no luck so far\.",
    r"I've checked for software updates, and my \{product_purchased\} is already running the latest version\.",
    r"I've reviewed the troubleshooting steps on the official support website, but they didn't resolve the problem\.",
    r"I've checked the device settings and made sure that everything is configured correctly\.",
    r"I've tried clearing the cache and data for the \{product_purchased\} app, but the issue persists\.",
]

COMPILED_TEMPLATES = [re.compile(t) for t in CLOSING_TEMPLATES]

# The generic, zero-information opener that precedes garbled filler in
# roughly two-thirds of rows. When a description starts with this exact
# opener, everything between it and the closing template is known junk
# (verified by inspection) and gets discarded. When a description does
# NOT start with this opener, the opening sentence(s) are typically a
# genuine, specific complaint (e.g. "I'm unable to access my {product}
# account. It keeps displaying an 'Invalid Credentials' error...") and
# must be kept, not discarded, even though a generic closing template
# still gets appended after them.
GENERIC_OPENER = re.compile(
    r"^I'm having an issue with the \{product_purchased\}\.\s*Please assist\.\s*",
    re.IGNORECASE,
)


def extract_clean_complaint(raw_description: str) -> tuple[str, str]:
    """
    Returns (cleaned_text, status). Status values:
      - "garbled_opener_stripped": generic "Please assist" opener detected,
        the (known-junk) middle discarded, closing template kept.
      - "opening_kept": no generic opener -- the opening sentence(s) were
        judged informative and kept, with the closing template appended.
      - "no_closing_template_found": no known closing template matched;
        text kept as-is (opener stripped if present) for manual review.
    """
    text = raw_description.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)

    opener_match = GENERIC_OPENER.match(text)
    had_generic_opener = opener_match is not None
    remainder = text[opener_match.end():] if had_generic_opener else text

    best_start = None
    for pattern in COMPILED_TEMPLATES:
        for m in pattern.finditer(remainder):
            if best_start is None or m.start() > best_start:
                best_start = m.start()

    if best_start is None:
        cleaned = remainder.strip() if remainder.strip() else text
        return cleaned, "no_closing_template_found"

    closing_part = remainder[best_start:].strip()
    opening_part = remainder[:best_start].strip()

    if had_generic_opener:
        # Known-junk middle -- discard it, keep only the closing template.
        return closing_part, "garbled_opener_stripped"

    if opening_part:
        # opening_part often ends with a trailing junk fragment glued onto
        # the last real sentence with no delimiter (e.g. "...urgently. If
        # your Data hasn't been copied and is no longer available, please
        # send us a message" -- everything after "urgently." is filler).
        # Truncate to the LAST proper sentence-ending punctuation mark
        # (., ?, or !) so only complete sentences survive; anything after
        # the final one (an incomplete trailing fragment) is dropped.
        #
        # Two patterns produce false sentence-boundary periods and need to
        # be excluded from candidates: abbreviation chains like "D.R.O.P."
        # (single letters separated by periods -- found as a whole span
        # first, since "D." alone doesn't look like an abbreviation until
        # the following "R.O.P." is seen too), and broken/truncated URLs
        # like "http://docs.google." -- both contain a "." that isn't
        # really an end of sentence.
        abbreviation_spans = [
            (m.start(), m.end())
            for m in re.finditer(r"(?:[A-Za-z]\.){2,}", opening_part)
        ]
        url_fragment = re.compile(r"(?:https?://|www\.)\S*$")
        # A period right after a digit is almost always a truncated number
        # (a version like "Ubuntu 24.10", a price, a quantity) rather than
        # a real sentence end -- exclude those candidates too.
        digit_before = re.compile(r"\d$")

        def inside_abbreviation(pos: int) -> bool:
            return any(start < pos <= end for start, end in abbreviation_spans)

        sentence_end_positions = []
        for m in re.finditer(r"[.?!]", opening_part):
            if inside_abbreviation(m.end()):
                continue
            if url_fragment.search(opening_part[:m.end()]):
                continue
            if digit_before.search(opening_part[:m.start()]):
                continue
            sentence_end_positions.append(m.end())

        # Some "complete" trailing sentences are still junk -- short,
        # throwaway interjections like "I'm sorry!" or "Thanks." that
        # happen to be grammatically valid. Real complaint sentences in
        # this dataset always run longer than a handful of words, so walk
        # backward from the last candidate and skip any final segment
        # under 5 words, stopping at the first substantial one.
        MIN_WORDS = 3
        while sentence_end_positions:
            end_pos = sentence_end_positions[-1]
            prev_pos = sentence_end_positions[-2] if len(sentence_end_positions) > 1 else 0
            segment = opening_part[prev_pos:end_pos].strip()
            if len(segment.split()) >= MIN_WORDS:
                break
            sentence_end_positions.pop()

        if sentence_end_positions:
            opening_part = opening_part[:sentence_end_positions[-1]].strip()
            cleaned = f"{opening_part} {closing_part}".strip()
        else:
            # No complete, trustworthy sentence in the opening -- nothing
            # safe to keep from it, fall back to the closing template only.
            cleaned = closing_part
        return cleaned, "opening_kept"
    return closing_part, "opening_kept"


def clean_file(path: str):
    df = pd.read_csv(path)

    if "cleaning_status" in df.columns:
        print(
            f"STOPPED: {path} already has a 'cleaning_status' column, which means "
            f"it's already been cleaned once. Running this script again on an "
            f"already-cleaned file corrupts the data (the {{product_purchased}} "
            f"placeholder is already substituted, so template matching breaks).\n"
            f"If you need to re-clean from scratch, restore the original raw CSV "
            f"first (e.g. re-run data/pull_new_tickets.py), then run this script once."
        )
        sys.exit(1)

    cleaned_descriptions = []
    statuses = []
    for _, row in df.iterrows():
        cleaned, status = extract_clean_complaint(row["description"])
        # Fill in the real product name and a generic stand-in for the
        # never-recorded error message.
        cleaned = cleaned.replace("{product_purchased}", row["product"])
        cleaned = cleaned.replace("{product_purchased_number}", "")
        cleaned = cleaned.replace("{error_message}", "an unspecified error")
        cleaned_descriptions.append(cleaned)
        statuses.append(status)

    df["raw_description"] = df["description"]
    df["description"] = cleaned_descriptions
    df["cleaning_status"] = statuses

    # Put cleaning_status right after description for easy scanning, keep
    # raw_description at the end as an audit trail rather than deleting it.
    cols = ["subject", "description", "cleaning_status", "product", "channel",
            "expected_category", "expected_priority", "expected_escalate", "raw_description"]
    df = df[cols]

    df.to_csv(path, index=False)

    n_stripped = sum(s == "garbled_opener_stripped" for s in statuses)
    n_kept = sum(s == "opening_kept" for s in statuses)
    n_unmatched = sum(s == "no_closing_template_found" for s in statuses)
    print(f"{path}: {n_stripped} rows had garbled opener stripped, "
          f"{n_kept} rows kept an informative opening sentence, "
          f"{n_unmatched} rows need a manual look (no closing template found)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data/clean_eval_descriptions.py <path_to_csv>")
        sys.exit(1)
    clean_file(sys.argv[1])