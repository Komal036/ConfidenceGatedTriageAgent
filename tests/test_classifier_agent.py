"""
Unit tests for app/agents/classifier.py.

Named test_classifier_agent.py (not test_classifier.py) to avoid colliding
with data/test_classifier.py, which is a standalone accuracy-eval script
against real labeled tickets, not a pytest unit test -- see README's
Evaluation Strategy section for that distinction.

These tests mock the Groq client entirely, so they run offline with no
API key or network call, and check classify_ticket()'s parsing and
fallback behavior in isolation from actual model quality (that's what
data/test_classifier.py is for).
"""
import json
from unittest.mock import patch, MagicMock

from app.agents.classifier import classify_ticket, VALID_CATEGORIES, VALID_PRIORITIES


def _mock_groq_response(content: str):
    """Builds a fake Groq response object shaped like the real SDK's."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    return mock_response


class TestValidResponses:
    @patch("app.agents.classifier.client")
    def test_valid_json_response_is_parsed(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps({"category": "Network", "priority": "High"})
        )
        result = classify_ticket("Wifi keeps dropping", "My connection cuts out every few minutes")
        assert result == {"category": "Network", "priority": "High"}

    @patch("app.agents.classifier.client")
    def test_strips_markdown_code_fences(self, mock_client):
        # Models sometimes wrap JSON in ```json ... ``` despite instructions
        # not to -- classify_ticket() strips this before parsing.
        fenced = "```json\n" + json.dumps({"category": "Billing", "priority": "Low"}) + "\n```"
        mock_client.chat.completions.create.return_value = _mock_groq_response(fenced)
        result = classify_ticket("Invoice question", "Why was I charged twice")
        assert result == {"category": "Billing", "priority": "Low"}


class TestInvalidValueFallback:
    @patch("app.agents.classifier.client")
    def test_unknown_category_falls_back_to_general_inquiry(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps({"category": "Not A Real Category", "priority": "Medium"})
        )
        result = classify_ticket("subject", "description")
        assert result["category"] == "General Inquiry"
        assert result["category"] in VALID_CATEGORIES

    @patch("app.agents.classifier.client")
    def test_unknown_priority_falls_back_to_medium(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            json.dumps({"category": "Hardware", "priority": "Urgent!!"})
        )
        result = classify_ticket("subject", "description")
        assert result["priority"] == "Medium"
        assert result["priority"] in VALID_PRIORITIES

    @patch("app.agents.classifier.client")
    def test_missing_fields_use_defaults(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(json.dumps({}))
        result = classify_ticket("subject", "description")
        assert result == {"category": "General Inquiry", "priority": "Medium"}


class TestMalformedOutputFailsSafe:
    @patch("app.agents.classifier.client")
    def test_unparseable_json_falls_back_to_safe_default(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_groq_response(
            "this is not json at all"
        )
        result = classify_ticket("subject", "description")
        # classify_ticket is documented to fail safe rather than crash --
        # this is the behavior the Escalation Judge's design assumes.
        assert result == {"category": "General Inquiry", "priority": "Medium"}

    @patch("app.agents.classifier.client")
    def test_empty_choices_falls_back_to_safe_default(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        result = classify_ticket("subject", "description")
        assert result == {"category": "General Inquiry", "priority": "Medium"}
