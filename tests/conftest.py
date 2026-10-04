"""Pytest fixtures.

Point the system at a throwaway data directory BEFORE any campus_safety module
is imported, so tests never touch the demo database you seed for the pitch.
"""

import os
import tempfile
from pathlib import Path

# Must happen before campus_safety.config is imported anywhere.
_TMP = tempfile.mkdtemp(prefix="campus_test_")
os.environ.setdefault("CAMPUS_DATA_DIR", _TMP)
os.environ.setdefault("CAMPUS_DETECTOR", "scripted")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_db():
    """Reset the schema before each test that uses the DB."""
    from campus_safety import db

    db.reset_db()
    yield


@pytest.fixture
def seeded():
    """A small deterministic set of incidents for API/history tests."""
    from campus_safety.zones import CAMPUS_ZONES
    from campus_safety import db

    db.upsert_zones(CAMPUS_ZONES)
    yield
