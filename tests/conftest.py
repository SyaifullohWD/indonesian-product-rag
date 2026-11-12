from __future__ import annotations

import os
import uuid

import pytest

from rag_slice.db import connect, init_db
from tests.support import FakeEmbedder, MappedEmbedder, OneHotEmbedder

__all__ = ["FakeEmbedder", "MappedEmbedder", "OneHotEmbedder"]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder(dim=1024)


@pytest.fixture
def db_url():
    return os.environ.get("RAG_DATABASE_URL", "postgresql:///product_rag?host=localhost&port=5433")


@pytest.fixture
def test_schema():
    return f"rag_slice_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def db_conn(db_url, test_schema):
    try:
        conn = connect(db_url)
    except Exception as e:
        pytest.skip(f"database not available: {e}")
    try:
        init_db(conn, schema=test_schema, dim=1024)
        yield conn
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{test_schema}" CASCADE')
            conn.commit()
        except Exception:
            pass
        conn.close()
