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
os.environ.setdefault("REDIS_URL", "")

import sys
import types
from unittest.mock import MagicMock
import numpy as np

_fake_ort = types.ModuleType("onnxruntime")

class _FakeSessionOptions:
    pass

class _FakeSession:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, output_names, input_feed, run_options=None):
        input_ids = input_feed["input_ids"]
        # If batch size is 1, it's the embedding model
        if input_ids.shape[0] == 1:
            seq_len = input_ids.shape[1]
            return [np.zeros((1, seq_len, 384), dtype=np.float32)]
        # Otherwise it's the cross encoder (batch size K)
        else:
            K = input_ids.shape[0]
            return [np.array([[0.9]] * K, dtype=np.float32)]

_fake_ort.InferenceSession = _FakeSession
_fake_ort.SessionOptions = _FakeSessionOptions

_fake_tokenizers = types.ModuleType("tokenizers")
class _FakeTokenizer:
    @classmethod
    def from_file(cls, *args, **kwargs):
        return cls()
    def enable_truncation(self, *args, **kwargs): pass
    def enable_padding(self, *args, **kwargs): pass
    def encode(self, text):
        m = MagicMock()
        m.ids = [0, 1, 2]
        m.attention_mask = [1, 1, 1]
        m.type_ids = [0, 0, 0]
        return m
    def encode_batch(self, texts):
        res = []
        for _ in texts:
            m = MagicMock()
            m.ids = [0, 1, 2]
            m.attention_mask = [1, 1, 1]
            m.type_ids = [0, 0, 0]
            res.append(m)
        return res

_fake_tokenizers.Tokenizer = _FakeTokenizer

_fake_huggingface_hub = types.ModuleType("huggingface_hub")
def _fake_hf_hub_download(*args, **kwargs):
    return "fake_path"
_fake_huggingface_hub.hf_hub_download = _fake_hf_hub_download

sys.modules.setdefault("onnxruntime", _fake_ort)
sys.modules.setdefault("tokenizers", _fake_tokenizers)
sys.modules.setdefault("huggingface_hub", _fake_huggingface_hub)

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
