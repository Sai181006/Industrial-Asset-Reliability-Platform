"""Database engine, session dependency, and declarative base."""

from __future__ import annotations

import os
from functools import wraps
from inspect import signature
from pathlib import Path
from time import sleep
from typing import Callable, ParamSpec, TypeVar

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_URL = f"sqlite:///{(ROOT / 'data' / 'interim' / 'reliability.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    pass


def _environment_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def build_engine(database_url: str):
    pool_size = _environment_integer("DB_POOL_SIZE", 5, 1, 1000)
    max_overflow = _environment_integer("DB_MAX_OVERFLOW", 10, 0, 1000)
    pool_timeout = _environment_integer("DB_POOL_TIMEOUT", 30, 1, 3600)
    pool_recycle = _environment_integer("DB_POOL_RECYCLE", 1800, -1, 604800)
    lock_timeout = _environment_integer("PG_LOCK_TIMEOUT_MS", 5000, 1, 2147483647)
    backend = make_url(database_url).get_backend_name()
    options = {"pool_pre_ping": True, "pool_recycle": pool_recycle}
    if backend != "sqlite" or make_url(database_url).database not in (None, "", ":memory:"):
        options.update(
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
        )
    configured_engine = create_engine(database_url, **options)
    if backend == "postgresql":
        @event.listens_for(configured_engine, "begin")
        def set_lock_timeout(connection):
            connection.exec_driver_sql(f"SET LOCAL lock_timeout = '{lock_timeout}ms'")
    return configured_engine


def lock_asset(db: Session, asset_id: str):
    from backend.app.models import Asset

    return db.scalar(
        select(Asset)
        .where(Asset.id == asset_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )


def is_duplicate_asset(error: IntegrityError, db: Session) -> bool:
    backend = db.get_bind().dialect.name
    if backend == "postgresql":
        return (
            getattr(error.orig, "sqlstate", None) == "23505"
            or getattr(error.orig, "pgcode", None) == "23505"
        ) and getattr(getattr(error.orig, "diag", None), "constraint_name", None) == "assets_pkey"
    if backend == "sqlite":
        return (
            getattr(error.orig, "sqlite_errorcode", None) in {1555, 2067}
            and str(error.orig) == "UNIQUE constraint failed: assets.id"
        )
    return False


def is_retryable_transaction_error(error: DBAPIError, db: Session) -> bool:
    return (
        db.get_bind().dialect.name == "postgresql"
        and not error.connection_invalidated
        and (getattr(error.orig, "sqlstate", None) or getattr(error.orig, "pgcode", None))
        in {"40001", "40P01", "55P03"}
    )


P = ParamSpec("P")
T = TypeVar("T")
TRANSACTION_ATTEMPTS = 3


def atomic_mutation(operation: Callable[P, T]) -> Callable[P, T]:
    operation_signature = signature(operation)

    @wraps(operation)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        db = operation_signature.bind(*args, **kwargs).arguments["db"]
        for attempt in range(TRANSACTION_ATTEMPTS):
            try:
                result = operation(*args, **kwargs)
                db.commit()
                return result
            except DBAPIError as error:
                db.rollback()
                if attempt + 1 == TRANSACTION_ATTEMPTS or not is_retryable_transaction_error(error, db):
                    raise
                sleep(0.05 * (2 ** attempt))
            except BaseException:
                db.rollback()
                raise
        raise RuntimeError("transaction attempts exhausted")

    return wrapped


engine = build_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    session = SessionLocal()
    try:
        yield session
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()

