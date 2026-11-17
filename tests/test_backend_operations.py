"""Health endpoint, fail-fast model loading, list limits and decision logging."""

import asyncio
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import backend.app.main as main
from backend.app.auth import authenticate
from backend.app.database import Base, get_db
from backend.app.decision import RiskThresholds, apply_decision
from backend.app.models import Alert, Asset, Prediction, WorkOrder


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    main.app.dependency_overrides[get_db] = override_get_db
    yield factory
    main.app.dependency_overrides.pop(get_db, None)
    engine.dispose()


def get(path: str, params: dict | None = None) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(path, params=params)

    return asyncio.run(send())


def test_health_is_public_and_reports_the_model(session_factory):
    main.app.dependency_overrides.pop(authenticate, None)  # no credentials at all
    assert get("/v1/assets").status_code == 401
    response = get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_id"] and body["model_version"]


def test_health_reports_unavailable_when_the_database_fails(session_factory):
    def broken_db():
        class Broken:
            def execute(self, *_args, **_kwargs):
                raise RuntimeError("database down")

        yield Broken()

    main.app.dependency_overrides[get_db] = broken_db
    response = get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_startup_fails_fast_when_the_model_cannot_load(monkeypatch):
    main.get_inference_service.cache_clear()

    def broken():
        raise RuntimeError("Model artifact checksum does not match manifest")

    monkeypatch.setattr(main, "create_inference_service", broken)
    try:
        with pytest.raises(RuntimeError, match="checksum"):
            with TestClient(main.app):
                pass
    finally:
        main.get_inference_service.cache_clear()


def seed_alerts(factory, count: int):
    with factory() as session:
        for index in range(1, count + 1):
            session.add(Asset(id=f"a{index}", name=f"Asset {index}"))
            session.add(
                Prediction(
                    id=index, asset_id=f"a{index}", window_start_cycle=1, window_end_cycle=30, raw_rul=5.0,
                    operational_rul=5.0, model_id="m", model_version="1",
                )
            )
        session.flush()
        for index in range(1, count + 1):
            session.add(Alert(id=index, asset_id=f"a{index}", prediction_id=index, severity="CRITICAL"))
        session.flush()
        for index in range(1, count + 1):
            session.add(WorkOrder(id=index, asset_id=f"a{index}", alert_id=index))
        session.commit()


@pytest.mark.parametrize("path", ["/v1/alerts", "/v1/work-orders"])
def test_workflow_lists_return_the_newest_items_up_to_the_limit(session_factory, path):
    seed_alerts(session_factory, 5)
    assert [row["id"] for row in get(path).json()] == [1, 2, 3, 4, 5]
    assert [row["id"] for row in get(path, {"limit": 2}).json()] == [4, 5]
    assert get(path, {"limit": 0}).status_code == 422
    assert get(path, {"limit": 1001}).status_code == 422


def test_decision_events_are_logged(session_factory, caplog):
    with session_factory() as session:
        session.add(Asset(id="x", name="X"))
        prediction = Prediction(
            asset_id="x", window_start_cycle=1, window_end_cycle=30, raw_rul=5.0, operational_rul=5.0,
            model_id="m", model_version="1",
        )
        session.add(prediction)
        session.flush()
        with caplog.at_level(logging.INFO, logger="reliability.decision"):
            apply_decision(session, prediction, RiskThresholds())
    messages = " | ".join(record.getMessage() for record in caplog.records)
    assert "alert opened asset=x severity=CRITICAL" in messages
    assert "work order created asset=x" in messages
