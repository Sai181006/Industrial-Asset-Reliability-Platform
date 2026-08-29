import asyncio
import hashlib
import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from backend.app.auth import Principal, Role, authenticate, configured_credentials, require_admin
from backend.app.main import app


def request(method: str, path: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


def test_authentication_accepts_configured_hash_and_rejects_bad_token(monkeypatch):
    token = "a-long-random-test-token"
    monkeypatch.setenv("AUTH_PRINCIPALS_JSON", json.dumps([{
        "subject": "operator@example.test", "role": "operator",
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
    }]))
    configured_credentials.cache_clear()
    principal = authenticate(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
    assert principal == Principal("operator@example.test", Role.OPERATOR)
    with pytest.raises(HTTPException) as exc:
        authenticate(HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong"))
    assert exc.value.status_code == 401
    configured_credentials.cache_clear()


def test_viewer_can_read_but_cannot_create_assets():
    app.dependency_overrides[authenticate] = lambda: Principal("test-viewer", Role.VIEWER)
    assert request("GET", "/v1/auth/me").json() == {"subject": "test-viewer", "role": "viewer"}
    response = request("POST", "/v1/assets", json={"asset_id": "forbidden", "name": "Forbidden"})
    assert response.status_code == 403
    assert response.json()["detail"] == "admin role required"


def test_operator_cannot_use_admin_dependency():
    with pytest.raises(HTTPException) as exc:
        require_admin(Principal("operator", Role.OPERATOR))
    assert exc.value.status_code == 403
