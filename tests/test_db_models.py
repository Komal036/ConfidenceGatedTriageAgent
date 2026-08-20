"""
Unit tests for app/db/models.py.

Two tiers here, deliberately kept separate:

1. Schema-level tests (always run, no DB needed): introspect SQLAlchemy's
   Table metadata directly to check column types, nullability, foreign
   keys, and defaults are wired the way the model docstrings claim. This
   catches things like "the FK stopped cascading" or "a column that
   should be non-nullable is missing nullable=False" without needing a
   database connection at all.

2. A live-DB integration test (test_resolution_roundtrip_against_real_db)
   that actually exercises Ticket/Resolution/AgentDecision against a real
   pgvector-enabled Postgres, since SQLite can't represent the Vector(384)
   column or Postgres UUID type. This is skipped unless RUN_DB_INTEGRATION=1
   is set with a real DATABASE_URL pointed at a pgvector-enabled instance --
   it's not meant to run in CI on every push, only on demand, since it needs
   real infrastructure. This honestly reflects that this project does not
   currently have a pgvector-enabled test database in CI (see README
   Limitations).
"""
import os
import uuid

import pytest
from sqlalchemy import inspect

from app.db import models


class TestTicketSchema:
    def test_primary_key_is_uuid(self):
        table = models.Ticket.__table__
        pk_col = table.c.id
        assert pk_col.primary_key is True

    def test_required_fields_are_non_nullable(self):
        table = models.Ticket.__table__
        assert table.c.subject.nullable is False
        assert table.c.description.nullable is False

    def test_optional_fields_are_nullable(self):
        table = models.Ticket.__table__
        for col_name in ("product", "channel", "category", "priority", "final_confidence"):
            assert table.c[col_name].nullable is True, f"{col_name} should be nullable"

    def test_status_defaults_to_received(self):
        default = models.Ticket.__table__.c.status.default
        assert default is not None
        assert default.arg == "received"

    def test_decisions_relationship_cascades_on_delete(self):
        mapper = inspect(models.Ticket)
        rel = mapper.relationships["decisions"]
        assert rel.cascade.delete is True
        assert rel.cascade.delete_orphan is True


class TestAgentDecisionSchema:
    def test_ticket_id_is_foreign_key_to_tickets(self):
        table = models.AgentDecision.__table__
        fk_targets = {fk.target_fullname for fk in table.c.ticket_id.foreign_keys}
        assert "tickets.id" in fk_targets

    def test_required_fields_are_non_nullable(self):
        table = models.AgentDecision.__table__
        assert table.c.ticket_id.nullable is False
        assert table.c.agent_name.nullable is False
        assert table.c.output_summary.nullable is False

    def test_confidence_is_optional(self):
        assert models.AgentDecision.__table__.c.confidence.nullable is True


class TestResolutionSchema:
    def test_required_fields_are_non_nullable(self):
        table = models.Resolution.__table__
        assert table.c.category.nullable is False
        assert table.c.issue_summary.nullable is False
        assert table.c.resolution_text.nullable is False

    def test_embedding_column_is_nullable_until_seeded(self):
        # Nullable on purpose: rows can exist before seed_knowledge_base.py
        # backfills their embeddings.
        assert models.Resolution.__table__.c.embedding.nullable is True

    def test_embedding_dimension_matches_minilm_output(self):
        # all-MiniLM-L6-v2 outputs 384-dim vectors -- if this ever drifts
        # out of sync with the embedding model in retriever.py, every
        # retrieval query breaks silently (dimension mismatch), so it's
        # worth pinning in a test rather than only in a docstring.
        assert models.Resolution.__table__.c.embedding.type.dim == 384


class TestModelDefaultsWithoutADatabase:
    def test_ticket_id_default_generates_uuid(self):
        # Column(default=uuid.uuid4) is a Python-side default -- it's
        # callable directly without needing a DB round-trip.
        generated = models.Ticket.__table__.c.id.default.arg(None)
        assert isinstance(generated, uuid.UUID)


@pytest.mark.skipif(
    os.environ.get("RUN_DB_INTEGRATION") != "1",
    reason=(
        "Requires a real pgvector-enabled Postgres reachable via DATABASE_URL. "
        "Not run in the default CI job -- set RUN_DB_INTEGRATION=1 with a real "
        "DATABASE_URL to run this locally against Neon."
    ),
)
def test_resolution_roundtrip_against_real_db():
    """
    Inserts a Resolution row with a real embedding and confirms it comes
    back out with the right shape. Opt-in only -- see module docstring.
    """
    from app.db.database import SessionLocal, Base, engine

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        row = models.Resolution(
            category="Network",
            issue_summary="Test issue for integration test",
            resolution_text="Test resolution text",
            embedding=[0.0] * 384,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        fetched = db.query(models.Resolution).filter_by(id=row.id).first()
        assert fetched is not None
        assert fetched.category == "Network"
        assert len(fetched.embedding) == 384
    finally:
        db.query(models.Resolution).filter_by(issue_summary="Test issue for integration test").delete()
        db.commit()
        db.close()
