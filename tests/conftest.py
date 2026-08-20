"""
Shared pytest fixtures.

Sets fake DATABASE_URL / GROQ_API_KEY env vars before any `app.*` module
gets imported anywhere in the test session. app/config.py's Settings()
is instantiated at import time and will crash on missing required fields
otherwise -- this has to happen before pytest collects any test module
that (transitively) imports app.config.
"""
import os

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

import sys
import types
from unittest.mock import MagicMock

# app/agents/retriever.py loads a real SentenceTransformer at *import time*
# (`_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")`), which
# downloads a model and pulls in torch. That's fine for the app itself, but
# it makes every test that merely imports app.main (even for routes that
# have nothing to do with retrieval) slow and network-dependent. Stub the
# whole sentence_transformers module before any app.* import happens, so
# retriever.py's module-level instantiation gets a lightweight fake instead.
_fake_st_module = types.ModuleType("sentence_transformers")


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text):
        # Real model outputs 384-dim vectors (matches models.py's
        # Vector(384) column) -- return a fixed-size fake so any code that
        # calls .tolist() on the result still works.
        class _FakeVector(list):
            def tolist(self):
                return list(self)

        return _FakeVector([0.0] * 384)


_fake_st_module.SentenceTransformer = _FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", _fake_st_module)

import pytest


@pytest.fixture
def mock_db_session():
    """
    A MagicMock standing in for a SQLAlchemy Session.

    Route tests use this instead of a live Postgres/pgvector connection --
    we're testing routing/serialization/status logic here, not the DB
    layer itself (see test_db_models.py for what IS tested against real
    SQLAlchemy machinery).
    """
    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock()
    return session
