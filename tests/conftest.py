"""Stable local test-process settings."""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import pytest

from backend.app.auth import Principal, Role, authenticate
from backend.app.main import app

@pytest.fixture(autouse=True)
def authenticated_admin():
    app.dependency_overrides[authenticate] = lambda: Principal("test-admin", Role.ADMIN)
    yield
    app.dependency_overrides.pop(authenticate, None)
