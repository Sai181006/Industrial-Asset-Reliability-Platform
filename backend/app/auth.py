"""Environment-configured bearer authentication and role authorization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
import hashlib
import hmac
import json
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class Role(IntEnum):
    VIEWER = 10
    OPERATOR = 20
    ADMIN = 30


@dataclass(frozen=True)
class Principal:
    subject: str
    role: Role

    @property
    def role_name(self) -> str:
        return self.role.name.lower()


@dataclass(frozen=True)
class Credential:
    token_sha256: str
    principal: Principal


bearer = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def configured_credentials() -> tuple[Credential, ...]:
    """Load hashed credentials without retaining bearer tokens in application state."""
    raw = os.getenv("AUTH_PRINCIPALS_JSON", "[]")
    try:
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError("expected a JSON array")
        credentials = []
        for record in records:
            subject = str(record["subject"]).strip()
            role = Role[str(record["role"]).upper()]
            digest = str(record["token_sha256"]).lower()
            if not subject or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("invalid subject or SHA-256 digest")
            credentials.append(Credential(digest, Principal(subject, role)))
        return tuple(credentials)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid AUTH_PRINCIPALS_JSON: {exc}") from exc


def authenticate(
    authorization: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> Principal:
    if authorization is None or authorization.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="bearer credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    supplied = hashlib.sha256(authorization.credentials.encode("utf-8")).hexdigest()
    for credential in configured_credentials():
        if hmac.compare_digest(supplied, credential.token_sha256):
            return credential.principal
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid bearer credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(minimum: Role):
    def dependency(principal: Principal = Depends(authenticate)) -> Principal:
        if principal.role < minimum:
            raise HTTPException(status_code=403, detail=f"{minimum.name.lower()} role required")
        return principal

    return dependency


require_operator = require_role(Role.OPERATOR)
require_admin = require_role(Role.ADMIN)
