import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
import httpx
import pytest
from sqlalchemy import create_engine, event, func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateSchema, DropSchema

from backend.app import main
from backend.app.database import Base, get_db
from backend.app.decision import RiskThresholds
from backend.app.models import Alert, Asset, Prediction, TelemetryPointRecord, WorkOrder
from backend.app.replay import replay_trajectory
from backend.app.schemas import RULPredictionResponse, TelemetryPoint


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL must explicitly select a PostgreSQL test database",
)


@pytest.fixture()
def postgres(monkeypatch):
    supplied_url = os.environ.get("TEST_DATABASE_URL")
    if not supplied_url:
        pytest.skip("TEST_DATABASE_URL is not set")
    url = make_url(supplied_url)
    if url.get_backend_name() != "postgresql":
        pytest.fail("TEST_DATABASE_URL must use PostgreSQL")
    schema = f"phase1_test_{uuid4().hex}"
    isolated_url = url.update_query_dict({
        "options": f"-csearch_path={schema} -clock_timeout=5000 -cstatement_timeout=15000",
        "connect_timeout": "5",
    })
    engine = create_engine(isolated_url, pool_size=8, max_overflow=0, pool_timeout=20)
    created = False
    try:
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
        created = True
        config = Config()
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.set_main_option(
            "sqlalchemy.url", isolated_url.render_as_string(hide_password=False).replace("%", "%%")
        )
        with monkeypatch.context() as migration_environment:
            migration_environment.delenv("DATABASE_URL", raising=False)
            command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT current_schema()")) == schema
            assert connection.scalar(text("SHOW search_path")) == schema
        factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
        state = SimpleNamespace(gate=None, remaining=0, pids=[], lock=Lock())

        def override_get_db():
            with factory() as session:
                with state.lock:
                    gate = state.gate if state.remaining else None
                    if gate is not None:
                        state.remaining -= 1
                if gate is not None:
                    pid = session.scalar(text("SELECT pg_backend_pid()"))
                    with state.lock:
                        state.pids.append(pid)
                    gate.wait(timeout=20)
                try:
                    yield session
                except BaseException:
                    session.rollback()
                    raise

        with monkeypatch.context() as dependencies:
            dependencies.setitem(main.app.dependency_overrides, get_db, override_get_db)
            dependencies.setitem(main.app.dependency_overrides, main.get_risk_thresholds, RiskThresholds)
            yield SimpleNamespace(engine=engine, factory=factory, schema=schema, config=config, state=state)
    finally:
        try:
            if created:
                assert schema.startswith("phase1_test_") and len(schema) == 44
                with engine.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True))
                with engine.connect() as connection:
                    assert connection.scalar(
                        text("SELECT count(*) FROM pg_namespace WHERE nspname = :schema"),
                        {"schema": schema},
                    ) == 0
        finally:
            engine.dispose()


class FakeInference:
    manifest = {"version": "phase1-fake-v1"}

    def __init__(self):
        self.calls = []
        self.lock = Lock()

    def predict(self, request):
        cycles = [point.cycle for point in request.telemetry]
        assert cycles == list(range(cycles[0], cycles[0] + 30))
        with self.lock:
            self.calls.append((request.asset_id, cycles[-1]))
        rul = request.telemetry[-1].sensor_2
        return RULPredictionResponse(
            asset_id=request.asset_id,
            window_start_cycle=cycles[0],
            window_end_cycle=cycles[-1],
            raw_rul=rul,
            operational_rul=max(rul, 0.0),
            model_id="phase1-fake",
            model_version=self.manifest["version"],
        )


@pytest.fixture()
def fake_inference(postgres, monkeypatch):
    service = FakeInference()
    monkeypatch.setitem(main.app.dependency_overrides, main.get_inference_service, lambda: service)
    return service


def point(cycle, rul=5.0):
    values = {name: float(cycle) for name in TelemetryPoint.model_fields if name != "cycle"}
    return {"cycle": cycle, **values, "sensor_2": rul}


def request(method, path, payload=None):
    async def send():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(send())


def ingest(asset_id, payload):
    return request("POST", f"/v1/assets/{asset_id}/telemetry", payload)


def seed(asset_id, cycles=29, rul=5.0):
    response = request("POST", "/v1/assets", {"asset_id": asset_id, "name": asset_id})
    assert response.status_code == 201, response.text
    body = None
    for cycle in range(1, cycles + 1):
        response = ingest(asset_id, point(cycle, rul))
        assert response.status_code == 200, response.text
        body = response.json()
    return body


@contextmanager
def independent_requests(postgres, count):
    state = postgres.state
    state.gate = Barrier(count)
    state.remaining = count
    state.pids = []
    try:
        yield
        assert state.remaining == 0
        assert len(state.pids) == count
        assert len(set(state.pids)) == count
    finally:
        state.gate.abort()
        state.gate = None
        state.remaining = 0


def race(postgres, *operations):
    with independent_requests(postgres, len(operations)):
        with ThreadPoolExecutor(max_workers=len(operations)) as executor:
            futures = [executor.submit(operation) for operation in operations]
            return [future.result(timeout=45) for future in futures]


def counts(postgres, asset_id):
    with postgres.factory() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model).where(model.asset_id == asset_id))
            for model in (TelemetryPointRecord, Prediction, Alert, WorkOrder)
        )


def test_migrations_match_metadata(postgres):
    with postgres.engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert set(context.get_current_heads()) == set(ScriptDirectory.from_config(postgres.config).get_heads())
        assert compare_metadata(context, Base.metadata) == []
        index = next(
            index for index in inspect(connection).get_indexes("alerts")
            if index["name"] == "uq_alert_one_active_per_asset"
        )
        assert index["unique"]
        assert index["column_names"] == ["asset_id"]
        predicate = str(index["dialect_options"]["postgresql_where"])
        assert predicate.replace("(", "").replace(")", "").strip() == "active = true"
        columns = {column["name"]: column for column in inspect(connection).get_columns("predictions")}
        assert columns["decision_processed"]["nullable"] is False
        assert columns["decision_processed"]["default"] == "false"


def test_duplicate_asset_race(postgres):
    insert_gate = Barrier(2)

    def synchronize_insert(session, flush_context, instances):
        if any(isinstance(record, Asset) for record in session.new):
            insert_gate.wait(timeout=20)

    event.listen(postgres.factory.class_, "before_flush", synchronize_insert)
    try:
        responses = race(postgres, *[
            lambda name=name: request("POST", "/v1/assets", {"asset_id": "duplicate", "name": name})
            for name in ("first", "second")
        ])
    finally:
        event.remove(postgres.factory.class_, "before_flush", synchronize_insert)
    assert sorted(response.status_code for response in responses) == [201, 409]
    winner = next(response.json() for response in responses if response.status_code == 201)
    with postgres.factory() as session:
        assert session.scalar(select(func.count()).select_from(Asset)) == 1
        assert session.get(Asset, "duplicate").name == winner["name"]


@pytest.mark.parametrize("conflicting", [False, True], ids=["identical", "conflicting"])
def test_telemetry_race(postgres, fake_inference, conflicting):
    seed("telemetry", cycles=0)
    payloads = [point(1), point(1, 25.0 if conflicting else 5.0)]
    responses = race(postgres, *[
        lambda payload=payload: ingest("telemetry", payload) for payload in payloads
    ])
    assert sorted(response.status_code for response in responses) == ([200, 409] if conflicting else [200, 200])
    assert sum(response.json().get("created", False) for response in responses) == 1
    winner = next(index for index, response in enumerate(responses) if response.json().get("created"))
    with postgres.factory() as session:
        record = session.scalars(select(TelemetryPointRecord)).one()
        for name, value in payloads[winner].items():
            assert getattr(record, name) == value
    assert counts(postgres, "telemetry") == (1, 0, 0, 0)
    assert fake_inference.calls == []


def test_thirtieth_cycle_race_has_one_prediction_alert_and_order(postgres, fake_inference):
    seed("thirtieth")
    responses = race(postgres, *[lambda: ingest("thirtieth", point(30)) for _ in range(4)])
    assert [response.status_code for response in responses] == [200] * 4
    bodies = [response.json() for response in responses]
    assert sum(body["created"] for body in bodies) == 1
    assert all(body["prediction"] == bodies[0]["prediction"] for body in bodies)
    assert all(body["decision"] == bodies[0]["decision"] for body in bodies)
    assert bodies[0]["prediction"]["window_start_cycle"] == 1
    assert bodies[0]["prediction"]["window_end_cycle"] == 30
    assert bodies[0]["decision"]["severity"] == "CRITICAL"
    assert counts(postgres, "thirtieth") == (30, 1, 1, 1)
    assert fake_inference.calls == [("thirtieth", 30)]
    with postgres.factory() as session:
        prediction = session.scalars(select(Prediction)).one()
        alert = session.scalars(select(Alert)).one()
        order = session.scalars(select(WorkOrder)).one()
        assert prediction.decision_processed
        assert prediction.decision_alert_id == alert.id == order.alert_id
        assert prediction.decision_work_order_id == order.id
        assert alert.prediction_id == prediction.id
        assert alert.active and alert.status == order.status == "OPEN"


def test_concurrent_decisions_escalate_without_downgrade(postgres, fake_inference):
    initial = seed("escalation", cycles=30, rul=50.0)
    alert_id = initial["decision"]["alert_id"]
    assert initial["decision"]["severity"] == "MEDIUM"
    responses = race(
        postgres,
        lambda: ingest("escalation", point(31, 5.0)),
        lambda: ingest("escalation", point(30, 50.0)),
    )
    assert [response.status_code for response in responses] == [200, 200]
    responses = race(postgres, *[lambda: ingest("escalation", point(32, 25.0)) for _ in range(2)])
    assert [response.status_code for response in responses] == [200, 200]
    assert all(response.json()["decision"]["severity"] == "HIGH" for response in responses)
    normal = ingest("escalation", point(33, 100.0))
    assert normal.status_code == 200
    assert normal.json()["decision"] == {"severity": "NORMAL", "alert_id": None, "work_order_id": None}
    assert counts(postgres, "escalation") == (33, 4, 1, 1)
    with postgres.factory() as session:
        alert = session.get(Alert, alert_id)
        assert alert.severity == "CRITICAL"
        assert alert.active
        assert session.get(Prediction, alert.prediction_id).window_end_cycle == 31
        assert session.scalars(select(WorkOrder)).one().alert_id == alert_id


def test_workflow_races_preserve_valid_states(postgres, fake_inference):
    decision = seed("workflow", cycles=30)["decision"]
    order_path = f"/v1/work-orders/{decision['work_order_id']}"
    alert_path = f"/v1/alerts/{decision['alert_id']}"
    responses = race(
        postgres,
        lambda: request("PATCH", order_path, {"status": "ASSIGNED", "assigned_to": "engineer"}),
        lambda: request("PATCH", order_path, {"status": "IN_PROGRESS"}),
    )
    assert responses[0].status_code == 200
    assert responses[1].status_code in (200, 409)
    order = request("GET", order_path).json()
    assert order["status"] == ("IN_PROGRESS" if responses[1].status_code == 200 else "ASSIGNED")
    assert order["assigned_to"] == "engineer"
    assert request("PATCH", order_path, {"status": "IN_PROGRESS"}).status_code == 200
    responses = race(
        postgres,
        lambda: request("PATCH", order_path, {"status": "RESOLVED"}),
        lambda: request("PATCH", order_path, {"status": "ASSIGNED", "assigned_to": "other"}),
    )
    assert [response.status_code for response in responses] == [200, 409]
    assert request("PATCH", alert_path, {"status": "ACKNOWLEDGED"}).status_code == 200
    responses = race(
        postgres,
        lambda: request("PATCH", alert_path, {"status": "RESOLVED"}),
        lambda: request("PATCH", alert_path, {"status": "OPEN"}),
    )
    assert [response.status_code for response in responses] == [200, 409]
    with postgres.factory() as session:
        order = session.get(WorkOrder, decision["work_order_id"])
        alert = session.get(Alert, decision["alert_id"])
        assert order.status == alert.status == "RESOLVED"
        assert order.assigned_to == "engineer"
        assert not alert.active
        assert order.resolved_at is not None and alert.resolved_at is not None
        assert order.last_updated_by == alert.last_updated_by == "test-admin"
    assert counts(postgres, "workflow") == (30, 1, 1, 1)


def test_alert_resolution_races_with_order_resolution(postgres, fake_inference):
    decision = seed("resolution", cycles=30)["decision"]
    order_path = f"/v1/work-orders/{decision['work_order_id']}"
    alert_path = f"/v1/alerts/{decision['alert_id']}"
    for payload in ({"status": "ASSIGNED", "assigned_to": "engineer"}, {"status": "IN_PROGRESS"}):
        assert request("PATCH", order_path, payload).status_code == 200
    assert request("PATCH", alert_path, {"status": "ACKNOWLEDGED"}).status_code == 200
    responses = race(
        postgres,
        lambda: request("PATCH", alert_path, {"status": "RESOLVED"}),
        lambda: request("PATCH", order_path, {"status": "RESOLVED"}),
    )
    assert responses[0].status_code in (200, 409)
    assert responses[1].status_code == 200
    with postgres.factory() as session:
        alert = session.get(Alert, decision["alert_id"])
        assert alert.status == ("RESOLVED" if responses[0].status_code == 200 else "ACKNOWLEDGED")
        assert alert.active == (responses[0].status_code == 409)
        assert session.get(WorkOrder, decision["work_order_id"]).status == "RESOLVED"
    assert request("PATCH", alert_path, {"status": "RESOLVED"}).status_code == 200


@pytest.mark.parametrize("failure_at", ["telemetry", "prediction", "decision"])
def test_ingestion_failure_rolls_back_all_flushed_records(postgres, fake_inference, monkeypatch, failure_at):
    seed("rollback")
    flushed = []

    def record_flush(session, flush_context):
        flushed.append(tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (TelemetryPointRecord, Prediction, Alert, WorkOrder)
        ))

    original_decision = main.apply_decision

    def fail_inference(payload):
        assert flushed[-1] == (30, 0, 0, 0)
        assert counts(postgres, "rollback") == (29, 0, 0, 0)
        raise RuntimeError("injected telemetry fault")

    def fail_decision(session, prediction, thresholds):
        assert prediction.id is not None
        assert flushed[-1] == (30, 1, 0, 0)
        if failure_at == "decision":
            original_decision(session, prediction, thresholds)
            assert flushed[-1] == (30, 1, 1, 1)
        assert counts(postgres, "rollback") == (29, 0, 0, 0)
        raise RuntimeError(f"injected {failure_at} fault")

    event.listen(postgres.factory.class_, "after_flush_postexec", record_flush)
    try:
        with monkeypatch.context() as fault:
            injected = Mock(side_effect=fail_inference if failure_at == "telemetry" else fail_decision)
            if failure_at == "telemetry":
                fault.setattr(fake_inference, "predict", injected)
            else:
                fault.setattr(main, "apply_decision", injected)
            with pytest.raises(RuntimeError, match=f"injected {failure_at} fault"):
                ingest("rollback", point(30))
            injected.assert_called_once()
    finally:
        event.remove(postgres.factory.class_, "after_flush_postexec", record_flush)
    assert counts(postgres, "rollback") == (29, 0, 0, 0)
    response = ingest("rollback", point(30))
    assert response.status_code == 200, response.text
    assert response.json()["created"]
    assert response.json()["decision"]["severity"] == "CRITICAL"
    assert counts(postgres, "rollback") == (30, 1, 1, 1)


def test_replay_retry_after_resolution_does_not_reopen(postgres, fake_inference):
    decision = seed("retry", cycles=30)["decision"]
    order_path = f"/v1/work-orders/{decision['work_order_id']}"
    alert_path = f"/v1/alerts/{decision['alert_id']}"
    for payload in (
        {"status": "ASSIGNED", "assigned_to": "engineer"},
        {"status": "IN_PROGRESS"},
        {"status": "RESOLVED"},
    ):
        assert request("PATCH", order_path, payload).status_code == 200
    for status in ("ACKNOWLEDGED", "RESOLVED"):
        assert request("PATCH", alert_path, {"status": status}).status_code == 200
    with postgres.factory() as session:
        timestamps = (
            session.get(Alert, decision["alert_id"]).resolved_at,
            session.get(WorkOrder, decision["work_order_id"]).resolved_at,
        )
    responses = race(postgres, *[lambda: ingest("retry", point(30)) for _ in range(3)])
    for response in responses:
        assert response.status_code == 200
        assert response.json()["created"] is False
        assert response.json()["decision"] == decision
    result = run_replays([("retry", [point(cycle) for cycle in range(1, 31)])], parallel=False, resume=False)[0]
    assert result.cycles_submitted == 30 and result.cycles_skipped == 0
    assert counts(postgres, "retry") == (30, 1, 1, 1)
    assert fake_inference.calls == [("retry", 30)]
    with postgres.factory() as session:
        alert = session.get(Alert, decision["alert_id"])
        order = session.get(WorkOrder, decision["work_order_id"])
        assert not alert.active
        assert alert.status == order.status == "RESOLVED"
        assert (alert.resolved_at, order.resolved_at) == timestamps


def run_replays(assets, parallel=True, resume=True):
    async def run():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            async def replay(asset_id, trajectory):
                return await replay_trajectory(
                    client, trajectory, asset_id=asset_id, asset_name=asset_id,
                    unit_id=81, cycles_per_second=0, resume=resume,
                )

            if parallel:
                return await asyncio.gather(*(replay(asset_id, trajectory) for asset_id, trajectory in assets))
            return [await replay(asset_id, trajectory) for asset_id, trajectory in assets]

    return asyncio.run(run())


def test_ordered_multiasset_concurrent_replay(postgres, fake_inference):
    assets = [(f"replay-{index}", [point(cycle, rul) for cycle in range(1, 36)])
              for index, rul in enumerate((5.0, 25.0, 100.0))]
    with independent_requests(postgres, len(assets)):
        results = run_replays(assets)
    for (asset_id, trajectory), result in zip(assets, results):
        assert result.status == "COMPLETED"
        assert result.asset_id == asset_id
        assert result.cycles_submitted == 35 and result.cycles_skipped == 0
        assert (result.first_cycle, result.last_cycle, result.predictions_received) == (1, 35, 6)
        rul = trajectory[-1]["sensor_2"]
        assert counts(postgres, asset_id) == (35, 6, int(rul <= 70), int(rul <= 10))
        with postgres.factory() as session:
            records = session.scalars(select(TelemetryPointRecord).where(
                TelemetryPointRecord.asset_id == asset_id
            ).order_by(TelemetryPointRecord.id)).all()
            assert [record.cycle for record in records] == list(range(1, 36))
            assert all(record.sensor_2 == rul for record in records)
            predictions = session.scalars(select(Prediction).where(
                Prediction.asset_id == asset_id
            ).order_by(Prediction.window_end_cycle)).all()
            assert [(prediction.window_start_cycle, prediction.window_end_cycle) for prediction in predictions] == [
                (cycle - 29, cycle) for cycle in range(30, 36)
            ]
            assert all(prediction.operational_rul == rul and prediction.decision_processed for prediction in predictions)
        assert [cycle for called_asset, cycle in fake_inference.calls if called_asset == asset_id] == list(range(30, 36))
    with independent_requests(postgres, len(assets)):
        resumed = run_replays(assets)
    assert all(result.cycles_submitted == 0 and result.cycles_skipped == 35 for result in resumed)
    assert len(fake_inference.calls) == 18


def test_real_artifact_sequential_and_parallel_replay(postgres, monkeypatch):
    from backend.app.inference import ARTIFACT_PATH, MANIFEST_PATH, RULInferenceService

    raw_dir = ROOT / "data" / "raw" / "source" / "CMAPSSData"
    required = [ARTIFACT_PATH, MANIFEST_PATH, *(raw_dir / name for name in (
        "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"
    ))]
    if not all(path.is_file() for path in required):
        pytest.skip("Promoted GRU artifact, manifest and official FD001 files are required")
    from backend.app.replay import load_training_trajectory

    service = RULInferenceService()
    monkeypatch.setitem(main.app.dependency_overrides, main.get_inference_service, lambda: service)
    trajectory = load_training_trajectory(raw_dir, 81)[:31]
    assert len(trajectory) == 31
    sequential = run_replays([("real-sequential", trajectory)], parallel=False)[0]
    with independent_requests(postgres, 2):
        parallel = run_replays([("real-parallel-1", trajectory), ("real-parallel-2", trajectory)])
    assert all(result.cycles_submitted == 31 and result.predictions_received == 2 for result in [sequential, *parallel])
    histories = []
    for asset_id in ("real-sequential", "real-parallel-1", "real-parallel-2"):
        assert counts(postgres, asset_id)[:2] == (31, 2)
        response = request("GET", f"/v1/assets/{asset_id}/predictions")
        assert response.status_code == 200
        history = response.json()
        assert [prediction["window_end_cycle"] for prediction in history] == [31, 30]
        assert all(prediction["model_id"] == service.manifest["model_id"] for prediction in history)
        assert all(prediction["model_version"] == service.manifest["version"] for prediction in history)
        assert all(prediction["operational_rul"] == max(prediction["raw_rul"], 0.0) for prediction in history)
        histories.append(history)
    for history in histories[1:]:
        for actual, expected in zip(history, histories[0]):
            assert actual["raw_rul"] == pytest.approx(expected["raw_rul"], rel=1e-6, abs=1e-6)
            assert actual["risk_severity"] == expected["risk_severity"]
