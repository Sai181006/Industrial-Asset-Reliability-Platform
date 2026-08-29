import asyncio
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, get_db
from backend.app.decision import RiskThresholds
from backend.app.main import app, get_risk_thresholds
from backend.app.models import Alert, Prediction, TelemetryPointRecord, WorkOrder
from ml.baseline import FEATURE_COLUMNS
from ml.data.cmapss import load_fd001


ROOT = Path(__file__).parents[1]
RAW_DIR = ROOT / "data" / "raw" / "source" / "CMAPSSData"


@pytest.fixture()
def database_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield factory
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="module")
def telemetry_points():
    if not RAW_DIR.exists():
        pytest.skip("Official FD001 files are not installed")
    train = load_fd001(RAW_DIR).train
    unit = train[train.unit_id == 81].sort_values("cycle").iloc[:31]
    points = []
    for _, row in unit.iterrows():
        point = {"cycle": int(row.cycle)}
        point.update({feature: float(row[feature]) for feature in FEATURE_COLUMNS})
        points.append(point)
    return points


@pytest.fixture(scope="module")
def full_trajectory():
    if not RAW_DIR.exists():
        pytest.skip("Official FD001 files are not installed")
    train = load_fd001(RAW_DIR).train
    unit = train[train.unit_id == 81].sort_values("cycle")
    points = []
    for _, row in unit.iterrows():
        point = {"cycle": int(row.cycle)}
        point.update({feature: float(row[feature]) for feature in FEATURE_COLUMNS})
        points.append(point)
    return points


def request(method: str, path: str, payload: dict) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(send())


def test_frontend_origin_is_allowed():
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.options(
                "/v1/assets",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )

    response = asyncio.run(send())
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


def test_incremental_ingestion_persists_one_idempotent_prediction(
    database_session_factory, telemetry_points
):
    created = request("POST", "/v1/assets", {"asset_id": "asset-81", "name": "Engine 81"})
    assert created.status_code == 201

    for point in telemetry_points[:29]:
        response = request("POST", "/v1/assets/asset-81/telemetry", point)
        assert response.status_code == 200
        assert response.json()["prediction"] is None

    thirtieth = request("POST", "/v1/assets/asset-81/telemetry", telemetry_points[29])
    assert thirtieth.status_code == 200
    body = thirtieth.json()
    assert body["created"] is True
    assert body["prediction"]["window_start_cycle"] == 1
    assert body["prediction"]["window_end_cycle"] == 30

    retry = request("POST", "/v1/assets/asset-81/telemetry", telemetry_points[29])
    assert retry.status_code == 200
    assert retry.json()["created"] is False
    assert retry.json()["prediction"] == body["prediction"]

    with database_session_factory() as session:
        telemetry_count = session.scalar(select(func.count()).select_from(TelemetryPointRecord))
        prediction_count = session.scalar(select(func.count()).select_from(Prediction))
    assert telemetry_count == 30
    assert prediction_count == 1


def test_conflicting_retry_is_rejected(database_session_factory, telemetry_points):
    request("POST", "/v1/assets", {"asset_id": "conflict", "name": "Conflict Engine"})
    assert request("POST", "/v1/assets/conflict/telemetry", telemetry_points[0]).status_code == 200
    changed = dict(telemetry_points[0])
    changed["sensor_2"] += 1.0
    response = request("POST", "/v1/assets/conflict/telemetry", changed)
    assert response.status_code == 409


def test_gap_prevents_window_inference(database_session_factory, telemetry_points):
    request("POST", "/v1/assets", {"asset_id": "gap", "name": "Gap Engine"})
    gapped = telemetry_points[:29] + [telemetry_points[30]]
    final_response = None
    for point in gapped:
        final_response = request("POST", "/v1/assets/gap/telemetry", point)
    assert final_response is not None
    assert final_response.json()["prediction"] is None
    with database_session_factory() as session:
        prediction_count = session.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.asset_id == "gap")
        )
    assert prediction_count == 0


def test_missing_asset_rejected(database_session_factory, telemetry_points):
    response = request("POST", "/v1/assets/missing/telemetry", telemetry_points[0])
    assert response.status_code == 404


def test_threshold_boundaries_and_validation():
    thresholds = RiskThresholds(critical=10, high=30, medium=70)
    assert thresholds.severity(10) == "CRITICAL"
    assert thresholds.severity(10.01) == "HIGH"
    assert thresholds.severity(30.01) == "MEDIUM"
    assert thresholds.severity(70.01) == "NORMAL"
    with pytest.raises(ValueError, match="thresholds"):
        RiskThresholds(critical=30, high=10, medium=70)


def test_critical_workflow_is_idempotent_and_transitions_are_guarded(
    database_session_factory, telemetry_points
):
    app.dependency_overrides[get_risk_thresholds] = lambda: RiskThresholds(
        critical=10_000, high=10_000, medium=10_000
    )
    try:
        request("POST", "/v1/assets", {"asset_id": "critical", "name": "Critical Engine"})
        final_response = None
        for point in telemetry_points[:30]:
            final_response = request("POST", "/v1/assets/critical/telemetry", point)
        assert final_response is not None
        decision = final_response.json()["decision"]
        assert decision["severity"] == "CRITICAL"
        assert decision["alert_id"] is not None
        assert decision["work_order_id"] is not None

        retry = request("POST", "/v1/assets/critical/telemetry", telemetry_points[29])
        assert retry.json()["decision"] == decision
        with database_session_factory() as session:
            assert session.scalar(select(func.count()).select_from(Alert)) == 1
            assert session.scalar(select(func.count()).select_from(WorkOrder)) == 1

        alert_id = decision["alert_id"]
        order_id = decision["work_order_id"]
        assert request("PATCH", f"/v1/work-orders/{order_id}", {"status": "IN_PROGRESS"}).status_code == 409
        assert request("PATCH", f"/v1/work-orders/{order_id}", {"status": "ASSIGNED"}).status_code == 422
        assigned = request(
            "PATCH",
            f"/v1/work-orders/{order_id}",
            {"status": "ASSIGNED", "assigned_to": "engineer-1"},
        )
        assert assigned.status_code == 200
        assert assigned.json()["last_updated_by"] == "test-admin"
        assert request("PATCH", f"/v1/work-orders/{order_id}", {"status": "IN_PROGRESS"}).status_code == 200

        assert request("PATCH", f"/v1/alerts/{alert_id}", {"status": "ACKNOWLEDGED"}).status_code == 200
        blocked = request("PATCH", f"/v1/alerts/{alert_id}", {"status": "RESOLVED"})
        assert blocked.status_code == 409
        assert request("PATCH", f"/v1/work-orders/{order_id}", {"status": "RESOLVED"}).status_code == 200
        resolved = request("PATCH", f"/v1/alerts/{alert_id}", {"status": "RESOLVED"})
        assert resolved.status_code == 200
        assert resolved.json()["active"] is False
        assert resolved.json()["last_updated_by"] == "test-admin"
    finally:
        app.dependency_overrides.pop(get_risk_thresholds, None)


def test_full_fd001_replay_reaches_readable_critical_workflow(
    database_session_factory, full_trajectory
):
    created = request("POST", "/v1/assets", {"asset_id": "replay-81", "name": "Replay Engine 81"})
    assert created.status_code == 201
    final = None
    for point in full_trajectory:
        final = request("POST", "/v1/assets/replay-81/telemetry", point)
        assert final.status_code == 200
    assert final is not None
    final_body = final.json()
    assert final_body["decision"]["severity"] == "CRITICAL"
    assert final_body["decision"]["work_order_id"] is not None

    detail = request("GET", "/v1/assets/replay-81", {}).json()
    assert detail["telemetry_count"] == len(full_trajectory)
    assert detail["prediction_count"] == len(full_trajectory) - 29
    assert detail["latest_cycle"] == full_trajectory[-1]["cycle"]
    assert detail["latest_prediction"]["risk_severity"] == "CRITICAL"
    assert detail["active_alert"]["severity"] == "CRITICAL"
    assert detail["active_work_order"]["status"] == "OPEN"

    recent = request("GET", "/v1/assets/replay-81/telemetry?limit=5", {}).json()
    assert [point["cycle"] for point in recent] == list(
        range(full_trajectory[-1]["cycle"] - 4, full_trajectory[-1]["cycle"] + 1)
    )
    history = request("GET", "/v1/assets/replay-81/predictions?limit=5", {}).json()
    assert len(history) == 5
    assert history[0]["window_end_cycle"] == full_trajectory[-1]["cycle"]

    alerts = request("GET", "/v1/alerts?asset_id=replay-81&active=true", {}).json()
    orders = request("GET", "/v1/work-orders?asset_id=replay-81&status=OPEN", {}).json()
    assert len(alerts) == 1
    assert len(orders) == 1
    assert request("GET", f"/v1/alerts/{alerts[0]['id']}", {}).status_code == 200
    assert request("GET", f"/v1/work-orders/{orders[0]['id']}", {}).status_code == 200
    assets = request("GET", "/v1/assets", {}).json()
    assert any(asset["asset_id"] == "replay-81" for asset in assets)
    renamed = request("PATCH", "/v1/assets/replay-81", {"name": "Renamed Replay Engine"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Renamed Replay Engine"
