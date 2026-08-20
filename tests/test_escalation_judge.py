"""
Unit tests for app/agents/escalation_judge.py.

judge_escalation() is a pure function (no LLM call, no DB call) so these
tests exercise its actual decision logic directly, rather than mocking
around it -- this is the one node in the pipeline that's cheap to test
exhaustively.

Covers all three escalation triggers named in the module docstring, plus
the "everything's fine" pass-through case, plus the boundary at exactly
ESCALATION_SIMILARITY_THRESHOLD.
"""
from app.agents.escalation_judge import judge_escalation, ESCALATION_SIMILARITY_THRESHOLD


class TestNoMatchEscalates:
    def test_no_match_found_escalates(self):
        result = judge_escalation(
            resolution_status="no_match",
            retrieved_match=None,
            priority="Medium",
        )
        assert result["escalate"] is True
        assert "no confident" in result["reason"].lower() or "no match" in result["reason"].lower() or "nothing to" in result["reason"].lower()

    def test_retrieved_match_none_escalates_even_if_status_resolved(self):
        # Defensive case: resolution_status says "resolved" but somehow
        # there's no match dict to back it up. Should still escalate --
        # the check is "or retrieved_match is None", not just status.
        result = judge_escalation(
            resolution_status="resolved",
            retrieved_match=None,
            priority="Medium",
        )
        assert result["escalate"] is True


class TestSimilarityThreshold:
    def test_similarity_below_threshold_escalates(self):
        result = judge_escalation(
            resolution_status="resolved",
            retrieved_match={"similarity": ESCALATION_SIMILARITY_THRESHOLD - 0.01},
            priority="Medium",
        )
        assert result["escalate"] is True
        assert "threshold" in result["reason"].lower()

    def test_similarity_at_exact_threshold_does_not_escalate(self):
        # judge_escalation uses a strict "<" comparison, so a match sitting
        # exactly on the threshold should count as confident enough.
        result = judge_escalation(
            resolution_status="resolved",
            retrieved_match={"similarity": ESCALATION_SIMILARITY_THRESHOLD},
            priority="Medium",
        )
        assert result["escalate"] is False

    def test_similarity_above_threshold_does_not_escalate(self):
        result = judge_escalation(
            resolution_status="resolved",
            retrieved_match={"similarity": 0.95},
            priority="Low",
        )
        assert result["escalate"] is False


class TestCriticalPriorityAlwaysEscalates:
    def test_critical_priority_escalates_despite_high_similarity(self):
        result = judge_escalation(
            resolution_status="resolved",
            retrieved_match={"similarity": 0.99},
            priority="Critical",
        )
        assert result["escalate"] is True
        assert "critical" in result["reason"].lower()

    def test_high_priority_does_not_force_escalation(self):
        # Only "Critical" is in ALWAYS_ESCALATE_PRIORITIES -- "High" should
        # be judged on match quality alone, same as Medium/Low.
        result = judge_escalation(
            resolution_status="resolved",
            retrieved_match={"similarity": 0.9},
            priority="High",
        )
        assert result["escalate"] is False


class TestHappyPath:
    def test_confident_match_non_critical_priority_does_not_escalate(self):
        result = judge_escalation(
            resolution_status="resolved",
            retrieved_match={"similarity": 0.85},
            priority="Low",
        )
        assert result["escalate"] is False
        assert "meets the confidence bar" in result["reason"]

    def test_reason_is_always_a_non_empty_string(self):
        # Every branch should return a human-readable reason -- this is
        # what gets written to the AgentDecision audit row, so an empty
        # reason would silently break the audit trail.
        cases = [
            ("no_match", None, "Medium"),
            ("resolved", {"similarity": 0.1}, "Low"),
            ("resolved", {"similarity": 0.9}, "Critical"),
            ("resolved", {"similarity": 0.9}, "Low"),
        ]
        for status, match, priority in cases:
            result = judge_escalation(status, match, priority)
            assert isinstance(result["reason"], str)
            assert len(result["reason"]) > 0
