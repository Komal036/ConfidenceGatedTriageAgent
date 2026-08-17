"""
Escalation Judge -- the fourth and final node in the pipeline.

Decides whether the Resolver's draft is safe to send automatically, or whether the ticket should go to a human instead. Deliberately rule-based
, not a weighted combination of
multiple signals and not a second LLM call asking for self-reported
confidence. See README's "Why confidence-gating instead of full automation?"
for the full reasoning -- in short: every signal used here already exists
elsewhere in the pipeline (Retriever's similarity score, Classifier's
priority, Resolver's resolution status), so there's nothing new to
calibrate or trust blindly, and with a 70-ticket eval set there isn't
enough data to responsibly tune a multi-signal weighted score.

Escalates when ANY of the following hold:
  1. resolution_status != "resolved" (the Retriever found no confident
     match, so the Resolver had nothing to draft from) -- there's nothing
     to ground an autonomous reply in.
  2. The Retriever's match similarity is below ESCALATION_SIMILARITY_THRESHOLD.
     This is deliberately a STRICTER, separate bar from the Retriever's own
     SIMILARITY_THRESHOLD (0.55, in retriever.py). The Retriever's threshold
     answers "is this a match at all?"; this threshold answers "is this
     match good enough to act on without a human?" -- two different
     questions, so two different thresholds. A match can clear the first
     bar and still fail the second (e.g. "wrong item billed"
     retrieval error at similarity 0.619 -- a real match, but not one this
     Judge should trust).
  3. The ticket's priority is "Critical". Regardless of match quality, a
     Critical ticket is judged too high-stakes to auto-resolve without a
     human in the loop -- the cost of being confidently wrong on a
     Critical ticket outweighs the convenience of automating it.

ESCALATION_SIMILARITY_THRESHOLD below is chosen from a real threshold sweep
against data/eval_tune.csv (see data/sweep_escalation_threshold.py), not
guessed -- per the project's "justify with data, not intuition" principle.

Sweep results (50 hand-labeled real Kaggle tickets, run against the actual
Classifier + Retriever pipeline):

  Threshold | Accuracy | False-Escalation Rate | False-Confidence Rate
  ----------|----------|------------------------|------------------------
  0.55      | 46.0%    | 76.7%                  | 20.0%
  0.60      | 48.0%    | 83.3%                  | 5.0%   <- chosen
  0.65      | 44.0%    | 93.3%                  | 0.0%
  0.70+     | 40.0%    | 100.0%                 | 0.0%   (flat -- see note)

0.60 was chosen over 0.65 despite 0.65's lower false-confidence rate,
because 0.65 sits above every real match similarity observed in this eval
set (max 0.678, but only a handful cleared even 0.65) -- at that point the
Judge is effectively just escalating almost everything, which trivially
drives false-confidence toward zero without reflecting a meaningfully
better-tuned policy. 0.60 is the actual accuracy peak (48%) and only trades
one additional false-confidence case (a difference of 1 ticket) for a
noticeably better false-escalation rate. Values 0.70 and above are flat
because no ticket in this eval set scored that high at all -- the ceiling
isn't threshold-driven at that point, it's a knowledge-base coverage limit
(see README's Results section: the KB currently matches ~24% of real
ticket phrasing after Week 3's expansion, up from 0% before it).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Data-driven value, chosen via a threshold sweep against data/eval_tune.csv
# (see data/sweep_escalation_threshold.py). Candidates 0.55-0.90 were tested;
# 0.60 was chosen as the accuracy peak (48%, the highest in the sweep),
# trading a small false-confidence rate (5.0%, ~1 ticket) for a meaningfully
# lower false-escalation rate than 0.65+. Thresholds at 0.65 and above drive
# false-confidence to 0% only because they exceed the highest similarity
# score observed on any real ticket in the eval set (0.678) -- at that point
# the Judge is just escalating everything, not making a tuned decision.
#
# Note the KB only produces a match at all for ~24% of real-world-phrased
# tickets right now (see data/diagnose_retrieval_scores.py findings) -- for
# the other ~76%, resolution_status != "resolved" already forces escalation
# regardless of this threshold. Expanding KB coverage further (see README
# Future Improvements) will matter more than re-tuning this number until
# that coverage gap narrows.
ESCALATION_SIMILARITY_THRESHOLD = 0.60

# Priorities that always escalate regardless of match confidence.
ALWAYS_ESCALATE_PRIORITIES = {"Critical"}


def judge_escalation(
    resolution_status: Optional[str],
    retrieved_match: Optional[dict],
    priority: Optional[str],
) -> dict:
    """
    Decides whether a ticket should be escalated to a human.

    Args:
        resolution_status: the Resolver's status, "resolved" or "no_match".
        retrieved_match: the Retriever's match dict (with a "similarity"
            key), or None if no confident match was found.
        priority: the Classifier's assigned priority.

    Returns:
        {
            "escalate": bool,
            "reason": str,  # human-readable justification, logged to AgentDecision
        }
    """
    if resolution_status != "resolved" or retrieved_match is None:
        return {
            "escalate": True,
            "reason": (
                "No confident knowledge base match found; nothing to "
                "ground an autonomous reply in."
            ),
        }

    similarity = retrieved_match.get("similarity")
    if similarity is not None and similarity < ESCALATION_SIMILARITY_THRESHOLD:
        return {
            "escalate": True,
            "reason": (
                f"Match similarity {similarity} is below the escalation "
                f"threshold ({ESCALATION_SIMILARITY_THRESHOLD}); too "
                f"uncertain to auto-resolve without a human check."
            ),
        }

    if priority in ALWAYS_ESCALATE_PRIORITIES:
        return {
            "escalate": True,
            "reason": (
                f"Priority is '{priority}'; Critical tickets always go to "
                f"a human regardless of match confidence."
            ),
        }

    return {
        "escalate": False,
        "reason": (
            f"Match similarity {similarity} meets the confidence bar "
            f"({ESCALATION_SIMILARITY_THRESHOLD}) and priority "
            f"('{priority}') is non-critical."
        ),
    }