"""
Unit tests for the FastAPI routes in app/main.py.

These mock the DB session (via FastAPI's dependency_overrides, the
documented way to swap get_db in tests) and app.main.run_triage_pipeline,
so no live Postgres/pgvector connection or Groq API key is needed. The
goal here is route-layer behavior: request validation, response shape,
and status-field logic -- not agent/model quality, which the eval scripts
in data/ already cover.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import get_db


def _pipeline_result(**overrides):
    """A default run_triage_pipeline() return value, override fields as needed."""
    result = {
        "category": "Network",
        "priority": "Medium",
        "retrieved_match": {
            "matched_issue": "WiFi drops intermittently",
            "resolution_text": "Restart the router and reconnect.",
            "category": "Network",
            "similarity": 0.82,
        },
        "resolution_status": "resolved",
        "draft_resolution": "Please restart your router and reconnect.",
        "tool_called": None,
        "tool_result": None,
        "escalate": False,
        "escalation_reason": "Match similarity meets the confidence bar.",
    }
    result.update(overrides)
    return result


@pytest.fixture
def client(mock_db_session):
    app.dependency_overrides[get_db] = lambda: mock_db_session
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealthCheck:
    def test_health_check_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSubmitTicketValidation:
    def test_missing_required_fields_returns_422(self, client):
        response = client.post("/submit-ticket", json={"subject": "Too short desc"})
        assert response.status_code == 422

    def test_subject_below_min_length_returns_422(self, client):
        response = client.post(
            "/submit-ticket",
            json={"subject": "hi", "description": "A valid description here"},
        )
        assert response.status_code == 422

    def test_description_below_min_length_returns_422(self, client):
        response = client.post(
            "/submit-ticket",
            json={"subject": "Valid subject", "description": "hi"},
        )
        assert response.status_code == 422

    def test_description_over_max_length_returns_422(self, client):
        response = client.post(
            "/submit-ticket",
            json={"subject": "Valid subject", "description": "x" * 1001},
        )
        assert response.status_code == 422


class TestSubmitTicketHappyPath:
    @patch("app.main.run_triage_pipeline")
    def test_resolved_ticket_returns_200_with_expected_shape(self, mock_pipeline, client):
        mock_pipeline.return_value = _pipeline_result()

        response = client.post(
            "/submit-ticket",
            json={
                "subject": "WiFi keeps dropping",
                "description": "My connection cuts out every few minutes at home.",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["category"] == "Network"
        assert body["priority"] == "Medium"
        assert body["status"] == "resolved"
        assert body["matched_issue"] == "WiFi drops intermittently"
        assert body["match_similarity"] == 0.82
        assert body["draft_resolution"] == "Please restart your router and reconnect."

    @patch("app.main.run_triage_pipeline")
    def test_optional_fields_are_accepted(self, mock_pipeline, client):
        mock_pipeline.return_value = _pipeline_result()

        response = client.post(
            "/submit-ticket",
            json={
                "subject": "WiFi keeps dropping",
                "description": "My connection cuts out every few minutes at home.",
                "product": "HomeRouter Pro",
                "channel": "email",
            },
        )
        assert response.status_code == 200


class TestSubmitTicketNoMatchPath:
    @patch("app.main.run_triage_pipeline")
    def test_no_match_ticket_has_no_match_status_and_null_resolution_fields(
        self, mock_pipeline, client
    ):
        mock_pipeline.return_value = _pipeline_result(
            retrieved_match=None,
            resolution_status="no_match",
            draft_resolution=None,
            tool_called=None,
            escalate=True,
            escalation_reason="No confident knowledge base match found.",
        )

        response = client.post(
            "/submit-ticket",
            json={
                "subject": "Something obscure and unmatched",
                "description": "A ticket the knowledge base has nothing close to.",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "no_match"
        assert body["matched_issue"] is None
        assert body["match_similarity"] is None
        assert body["draft_resolution"] is None

    @patch("app.main.run_triage_pipeline")
    def test_db_session_is_committed_after_pipeline_runs(self, mock_pipeline, client, mock_db_session):
        # Regression guard: submit_ticket writes the initial Ticket row,
        # then updates it after the pipeline runs, then adds AgentDecision
        # rows -- each stage should commit. This doesn't assert an exact
        # count (that's an implementation detail likely to shift), just
        # that at least one commit happened after a successful submission.
        mock_pipeline.return_value = _pipeline_result()
        client.post(
            "/submit-ticket",
            json={"subject": "WiFi keeps dropping", "description": "Connection cuts out often."},
        )
        assert mock_db_session.commit.called


class TestEscalationFieldsInResponse:
    """
    Regression tests for a bug where TicketResponse never carried the
    Judge's escalate/escalation_reason decision to the client at all.

    The frontend's TicketResult type declared `escalated: boolean` and
    `escalation_reason: string` as if the backend sent them, but
    submitTicket() just does an untyped `res.json()` -- so at runtime
    `result.escalated` was always `undefined`, and
    `!result.escalated` (used to compute the Confidence Gate's `open`
    prop) evaluated to `true` for every ticket regardless of the real
    decision. The UI showed "OPEN — AUTO-RESOLVED" unconditionally, even
    for tickets the backend had genuinely escalated.

    These tests pin the fix at the API boundary: the response body must
    actually contain a boolean `escalated` and a non-empty
    `escalation_reason` string that reflect judge_escalation()'s real
    output -- not just that the pipeline *computed* them internally
    (that's already covered by test_escalation_judge.py), but that
    submit_ticket() actually puts them on TicketResponse.
    """

    @patch("app.main.run_triage_pipeline")
    def test_no_match_ticket_response_is_marked_escalated(self, mock_pipeline, client):
        mock_pipeline.return_value = _pipeline_result(
            retrieved_match=None,
            resolution_status="no_match",
            draft_resolution=None,
            tool_called=None,
            escalate=True,
            escalation_reason=(
                "No confident knowledge base match found; nothing to "
                "ground an autonomous reply in."
            ),
        )

        response = client.post(
            "/submit-ticket",
            json={
                "subject": "Intermittent 3D rendering artifacts in CAD export",
                "description": (
                    "When I export assemblies larger than 200 parts to STEP "
                    "format, random polygon glitches appear on curved "
                    "surfaces only in the exported file, not in the live "
                    "viewport. Happens on both my workstations."
                ),
            },
        )

        assert response.status_code == 200
        body = response.json()

        # The two fields the frontend bug silently depended on but never
        # actually received -- assert both the field's presence/type and
        # its value, so a schema that merely declares `escalated: bool`
        # without main.py ever setting it would still fail this test.
        assert "escalated" in body
        assert body["escalated"] is True

        assert "escalation_reason" in body
        assert isinstance(body["escalation_reason"], str)
        assert len(body["escalation_reason"]) > 0
        assert body["escalation_reason"] == (
            "No confident knowledge base match found; nothing to "
            "ground an autonomous reply in."
        )

    @patch("app.main.run_triage_pipeline")
    def test_resolved_confident_ticket_response_is_not_escalated(self, mock_pipeline, client):
        # The flip side of the bug: since the old frontend logic happened
        # to default to "open" for everything, a naive fix could
        # overcorrect and mark every ticket as escalated instead. Pin the
        # non-escalated case too so both directions are covered.
        mock_pipeline.return_value = _pipeline_result(
            escalate=False,
            escalation_reason=(
                "Match similarity 0.82 meets the confidence bar (0.6) and "
                "priority ('Medium') is non-critical."
            ),
        )

        response = client.post(
            "/submit-ticket",
            json={
                "subject": "WiFi keeps dropping",
                "description": "My connection cuts out every few minutes at home.",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["escalated"] is False
        assert len(body["escalation_reason"]) > 0

    @patch("app.main.run_triage_pipeline")
    def test_critical_priority_ticket_is_escalated_despite_a_match(self, mock_pipeline, client):
        # Covers the third escalation trigger (Critical priority always
        # escalates, per escalation_judge.py) at the API-response level,
        # not just the unit level -- a match existing shouldn't leak
        # through as escalated=False just because retrieved_match is set.
        mock_pipeline.return_value = _pipeline_result(
            priority="Critical",
            escalate=True,
            escalation_reason=(
                "Priority is 'Critical'; Critical tickets always go to a "
                "human regardless of match confidence."
            ),
        )

        response = client.post(
            "/submit-ticket",
            json={
                "subject": "Cannot log in, account shows suspicious activity",
                "description": "Locked out completely with a security alert, blocking my whole team.",
            },
        )

        body = response.json()
        assert body["priority"] == "Critical"
        assert body["matched_issue"] is not None  # a match was found...
        assert body["escalated"] is True  # ...but Critical overrides it anyway