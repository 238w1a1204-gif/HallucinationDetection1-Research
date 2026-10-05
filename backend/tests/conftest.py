"""
Shared pytest fixtures.

Uses an in-memory SQLite database so tests never touch the real
hallucination_detection.db on disk.

Key design decisions:
- A single shared in-memory engine is used for the whole test session.
  SQLite :memory: databases are connection-local; we use
  StaticPool so every call to connect() reuses the same connection,
  making the in-memory DB persistent across the session.
- Tables are created once and truncated (not dropped) between tests for
  isolation without losing the schema.
- The startup `init_db()` call in main.py is patched to a no-op so it
  doesn't touch the production DB path.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch

from app.db.database import Base, get_db
from app.main import app

# ── Shared in-memory engine ────────────────────────────────────────────────
_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,           # single connection, shared across threads
)
_TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

# Create all tables once when the test module is imported.
Base.metadata.create_all(bind=_engine)


def override_get_db():
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


def _clear_tables():
    """Delete all rows from every table without dropping the schema."""
    with _engine.connect() as conn:
        # Disable FK constraints so we can truncate in any order.
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()


@pytest.fixture()
def client():
    """
    A FastAPI TestClient backed by the shared in-memory database.
    Tables are cleared before each test so tests are fully isolated.
    The startup init_db() is patched to a no-op (tables are already created).
    """
    _clear_tables()
    with patch("app.main.init_db"):          # prevent prod-DB init during lifespan
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
