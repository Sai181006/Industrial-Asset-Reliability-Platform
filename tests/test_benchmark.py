from contextlib import contextmanager
import json
import os
from types import SimpleNamespace

import httpx
import pytest

from scripts import benchmark_postgres as benchmark


def decision_history():
    from backend.app.decision import RiskThresholds

    thresholds = RiskThresholds()
    predictions = []
    responses = {}
    critical_seen = False
    for cycle, rul in enumerate((100, 50, 25, 5, 100, 50, 5), 30):
        severity = thresholds.severity(rul)
        critical_seen = critical_seen or severity == "CRITICAL"
        alert_id = 10 if severity != "NORMAL" else None
        order_id = 20 if critical_seen and severity != "NORMAL" else None
        prediction = SimpleNamespace(id=cycle, asset_id="bench-0", window_end_cycle=cycle,
                                     operational_rul=rul, risk_severity=severity, decision_processed=True,
                                     decision_alert_id=alert_id, decision_work_order_id=order_id)
        predictions.append(prediction)
        responses[cycle] = {"severity": severity, "alert_id": alert_id, "work_order_id": order_id}
    alert = SimpleNamespace(id=10, asset_id="bench-0", prediction_id=33, severity="CRITICAL",
                            active=True, status="OPEN", resolved_at=None)
    order = SimpleNamespace(id=20, asset_id="bench-0", alert_id=10, status="OPEN", resolved_at=None)
    return predictions, [alert], [order], responses, thresholds


def test_decision_escalation_and_recovery_history():
    assert benchmark.validate_decisions("bench-0", *decision_history()) == ([], [])


@pytest.mark.parametrize("target,field,value", [
    ("alert", "prediction_id", 36), ("alert", "prediction_id", None),
    ("alert", "severity", "MEDIUM"), ("alert", "active", False),
    ("alert", "status", "RESOLVED"), ("alert", "asset_id", "other"),
    ("order", "alert_id", None), ("order", "alert_id", 99),
    ("order", "asset_id", "other"), ("order", "status", "RESOLVED"),
    ("prediction", "decision_alert_id", None), ("prediction", "decision_work_order_id", None),
])
def test_count_preserving_decision_state_faults(target, field, value):
    predictions, alerts, orders, responses, thresholds = decision_history()
    record = {"alert": alerts[0], "order": orders[0], "prediction": predictions[3]}[target]
    setattr(record, field, value)
    mismatches, errors = benchmark.validate_decisions("bench-0", predictions, alerts, orders, responses, thresholds)
    assert mismatches
    assert (len(predictions), len(alerts), len(orders)) == (7, 1, 1)


@pytest.mark.parametrize("cycle,field,value", [
    (30, "decision_alert_id", 10), (31, "decision_work_order_id", 20),
    (34, "decision_work_order_id", 20), (35, "decision_work_order_id", None),
])
def test_historical_decision_ids_not_final_workflow_ids(cycle, field, value):
    predictions, alerts, orders, responses, thresholds = decision_history()
    setattr(predictions[cycle - 30], field, value)
    mismatches, errors = benchmark.validate_decisions("bench-0", predictions, alerts, orders, responses, thresholds)
    assert mismatches and errors


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="explicit local TEST_DATABASE_URL required")
@pytest.mark.parametrize("fault", [
    "response_null", "response_severity", "response_alert", "response_order", "response_null_ids",
    "prediction_alert_null", "prediction_alert_wrong", "prediction_order_null", "prediction_order_wrong",
    "alert_prediction_wrong", "alert_escalation", "order_alert_wrong",
])
def test_postgres_count_preserving_decision_faults(monkeypatch, fault):
    from backend.app import auth, main
    from backend.app.models import Alert, Prediction, WorkOrder
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    for name in ("RUL_CRITICAL_THRESHOLD", "RUL_HIGH_THRESHOLD", "RUL_MEDIUM_THRESHOLD"):
        monkeypatch.setenv(name, "1000000")
    original_environment = dict(os.environ)
    original_overrides = dict(main.app.dependency_overrides)
    original_credentials = auth.configured_credentials
    original_database = benchmark.isolated_database
    original_post = httpx.AsyncClient.post
    state = {}

    @contextmanager
    def capture_database(url, report):
        with original_database(url, report) as engine:
            state["engine"] = engine
            yield engine

    async def corrupt(client, path, **kwargs):
        response = await original_post(client, path, **kwargs)
        if path != "/v1/assets/bench-1/telemetry" or kwargs["json"]["cycle"] != 31:
            return response
        assert response.status_code == 200
        state["injected"] = True
        if fault.startswith("response_"):
            body = response.json()
            if fault == "response_null":
                body["decision"] = None
            elif fault == "response_severity":
                body["decision"]["severity"] = "NORMAL"
            elif fault == "response_alert":
                body["decision"]["alert_id"] = -1
            elif fault == "response_order":
                body["decision"]["work_order_id"] = -1
            else:
                body["decision"]["alert_id"] = None
                body["decision"]["work_order_id"] = None
            return httpx.Response(200, json=body)
        with Session(state["engine"]) as session:
            prediction = session.scalar(select(Prediction).where(
                Prediction.asset_id == "bench-1", Prediction.window_end_cycle == 31))
            other_prediction = session.scalar(select(Prediction).where(Prediction.asset_id == "bench-0"))
            alert = session.scalar(select(Alert).where(Alert.asset_id == "bench-1"))
            other_alert = session.scalar(select(Alert).where(Alert.asset_id == "bench-0"))
            other_order = session.scalar(select(WorkOrder).where(WorkOrder.asset_id == "bench-0"))
            if fault == "prediction_alert_null":
                prediction.decision_alert_id = None
            elif fault == "prediction_alert_wrong":
                prediction.decision_alert_id = other_alert.id
            elif fault == "prediction_order_null":
                prediction.decision_work_order_id = None
            elif fault == "prediction_order_wrong":
                prediction.decision_work_order_id = other_order.id
            elif fault == "alert_prediction_wrong":
                alert.prediction_id = other_prediction.id
            elif fault == "alert_escalation":
                alert.severity = "MEDIUM"
            else:
                order = session.scalar(select(WorkOrder).where(WorkOrder.asset_id == "bench-1"))
                order.asset_id, other_order.asset_id = other_order.asset_id, order.asset_id
            session.commit()
        return response

    monkeypatch.setattr(benchmark, "isolated_database", capture_database)
    monkeypatch.setattr(httpx.AsyncClient, "post", corrupt)
    report = benchmark.run(benchmark.parse_args(["--assets", "2", "--cycles", "31", "--concurrency", "1"]))
    assert state["injected"]
    assert report["status"] == "failed"
    assert report["persisted"]["actual"] == report["persisted"]["expected"]
    assert all(row["actual"] == row["expected"] for row in report["persisted"]["per_asset"].values())
    if fault.startswith("response_"):
        assert report["metrics"]["errors"] == 1
        assert report["request_errors"] == [{"asset": "bench-1", "cycle": 31, "kind": "decision_response_mismatch"}]
    else:
        assert report["validation"]["mismatches"]
    assert report["cleanup"]["schema_dropped"]
    assert dict(os.environ) == original_environment
    assert main.app.dependency_overrides == original_overrides
    assert auth.configured_credentials is original_credentials


def test_cli_defaults():
    args = benchmark.parse_args([])
    assert (args.assets, args.cycles, args.concurrency, args.unit_id) == (4, 60, 4, 81)


@pytest.mark.parametrize("flag,value", [
    ("--assets", "0"), ("--assets", "17"), ("--assets", "1.5"),
    ("--concurrency", "0"), ("--concurrency", "17"),
    ("--cycles", "0"), ("--cycles", "363"), ("--cycles", "-1"),
    ("--unit-id", "0"), ("--unit-id", "101"),
    ("--base-url", "http://example.invalid"), ("--database-url", "postgresql://remote/db"),
])
def test_cli_rejects_out_of_bounds_and_targets(flag, value):
    with pytest.raises(ValueError):
        benchmark.parse_args([flag, value])


@pytest.mark.parametrize("assets,concurrency,cycles", [(1, 1, 1), (16, 16, 240)])
def test_workload_boundaries(assets, concurrency, cycles):
    args = benchmark.parse_args(["--assets", str(assets), "--concurrency", str(concurrency), "--cycles", str(cycles)])
    benchmark.validate_workload(args, [{"cycle": cycle} for cycle in range(1, 241)])


@pytest.mark.parametrize("changes", [
    {"cycles": 241}, {"cycles": 0}, {"assets": 17}, {"concurrency": 17}, {"unit_id": 101},
])
def test_programmatic_bounds(changes):
    values = vars(benchmark.parse_args([])) | changes
    with pytest.raises(ValueError):
        benchmark.validate_workload(SimpleNamespace(**values), [{"cycle": cycle} for cycle in range(1, 241)])


def test_rejects_unordered_source():
    with pytest.raises(ValueError):
        benchmark.validate_workload(benchmark.parse_args(["--cycles", "2"]), [{"cycle": 2}, {"cycle": 1}])


@pytest.mark.parametrize("url", [
    None, "sqlite:///:memory:", "postgresql://user:secret@192.0.2.1/db",
    "postgresql://user:secret@example.invalid/db", "postgresql://user:secret@/db",
    "postgresql://user:secret@127.0.0.1/db?host=192.0.2.1",
    "postgresql://user:secret@127.0.0.1/db?service=remote",
    "postgresql://user:secret@127.0.0.1/db?options=-csearch_path=public",
    "postgresql://user:secret@[::ffff:192.0.2.1]/db",
    "postgresql://user:secret@127.0.0.1:0/db",
])
def test_database_target_is_local_only(url):
    with pytest.raises(ValueError) as error:
        benchmark.local_database_url(url)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("host,expected", [("localhost", "127.0.0.1"), ("127.0.0.1", "127.0.0.1"), ("[::1]", "::1")])
def test_local_database_target(host, expected):
    url = benchmark.local_database_url(f"postgresql://user:secret@{host}:5433/reliability")
    assert url.host == expected
    assert url.drivername == "postgresql+psycopg"
    assert url.port == 5433


def test_metric_computation():
    samples = [40.0, 10.0, 30.0, 20.0]
    result = benchmark.compute_metrics(samples, 2.0, errors=1)
    assert result["telemetry_requests"] == 4
    assert result["successful_requests"] == 3
    assert result["errors"] == 1
    assert result["latency_ms"] == {"p50": 25.0, "p95": 38.5, "max": 40.0}
    assert result["throughput_requests_per_second"] == 2.0
    assert result["successful_requests_per_second"] == 1.5
    assert result["elapsed_seconds"] == 2.0
    assert samples == [40.0, 10.0, 30.0, 20.0]


def test_empty_and_singleton_metrics():
    assert benchmark.compute_metrics([], 1.0)["latency_ms"] == {"p50": None, "p95": None, "max": None}
    assert benchmark.compute_metrics([7.0], 1.0)["latency_ms"] == {"p50": 7.0, "p95": 7.0, "max": 7.0}


@pytest.mark.parametrize("samples,elapsed,errors", [
    ([1], 0, 0), ([1], -1, 0), ([1], float("nan"), 0), ([1], float("inf"), 0),
    ([-1], 1, 0), ([float("nan")], 1, 0), ([float("inf")], 1, 0),
    ([1], 1, -1), ([1], 1, 2), ([1], 1, 0.5),
])
def test_invalid_metrics(samples, elapsed, errors):
    with pytest.raises(ValueError):
        benchmark.compute_metrics(samples, elapsed, errors)


def test_missing_database_url_fails_json_only(monkeypatch, capsys):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    assert benchmark.main([]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "ValueError"


def test_failure_does_not_disclose_exception(monkeypatch, capsys):
    def fail(args, report):
        raise RuntimeError("postgresql://user:super-secret@127.0.0.1/private")

    monkeypatch.setattr(benchmark, "run", fail)
    assert benchmark.main([]) == 1
    captured = capsys.readouterr()
    assert "super-secret" not in captured.out + captured.err
    assert json.loads(captured.out)["status"] == "failed"


def test_invalid_cli_does_not_echo_secrets(capsys):
    assert benchmark.main(["--base-url", "http://user:super-secret@example.invalid"]) == 1
    captured = capsys.readouterr()
    assert "super-secret" not in captured.out + captured.err
    assert json.loads(captured.out)["status"] == "failed"


def test_failed_validation_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "run", lambda args, report: report.update(status="failed", validation={"passed": False}))
    assert benchmark.main([]) == 1
    assert json.loads(capsys.readouterr().out)["validation"]["passed"] is False


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="explicit local TEST_DATABASE_URL required")
def test_local_postgres_benchmark_and_cleanup(monkeypatch):
    from backend.app import auth, main
    from backend.app.database import build_engine
    from sqlalchemy import text

    benchmark.local_database_url(os.environ["TEST_DATABASE_URL"])
    original_environment = dict(os.environ)
    original_overrides = dict(main.app.dependency_overrides)
    original_credentials = auth.configured_credentials
    report = benchmark.run(benchmark.parse_args(["--assets", "2", "--cycles", "31", "--concurrency", "2"]))
    assert report["status"] == "passed"
    assert report["auth"]["bypass"] is False
    assert report["metrics"]["telemetry_requests"] == 62
    assert report["metrics"]["errors"] == 0
    assert report["persisted"]["actual"]["predictions"] == 4
    assert report["cleanup"] == {"schema_created": True, "schema_dropped": True}
    assert dict(os.environ) == original_environment
    assert main.app.dependency_overrides == original_overrides
    assert auth.configured_credentials is original_credentials

    original_post = httpx.AsyncClient.post

    async def rejected_telemetry(client, path, **kwargs):
        if path.endswith("/telemetry"):
            return httpx.Response(500)
        return await original_post(client, path, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(httpx.AsyncClient, "post", rejected_telemetry)
        rejected = benchmark.run(benchmark.parse_args(["--assets", "1", "--cycles", "1"]))
    assert rejected["status"] == "failed"
    assert rejected["metrics"]["errors"] == 1
    assert rejected["validation"]["mismatches"]
    assert rejected["cleanup"]["schema_dropped"]
    assert auth.configured_credentials is original_credentials
    assert main.app.dependency_overrides == original_overrides
    assert dict(os.environ) == original_environment

    async def fail(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(benchmark, "measure", fail)
    failed = {}
    with pytest.raises(RuntimeError, match="injected failure"):
        benchmark.run(benchmark.parse_args([]), failed)
    assert failed["cleanup"] == {"schema_created": True, "schema_dropped": True}
    assert dict(os.environ) == original_environment
    assert main.app.dependency_overrides == original_overrides
    engine = build_engine(benchmark.local_database_url(os.environ["TEST_DATABASE_URL"]))
    try:
        with engine.connect() as connection:
            for schema in (report["schema"], rejected["schema"], failed["schema"]):
                assert connection.scalar(text("SELECT count(*) FROM pg_namespace WHERE nspname=:schema"), {"schema": schema}) == 0
    finally:
        engine.dispose()
