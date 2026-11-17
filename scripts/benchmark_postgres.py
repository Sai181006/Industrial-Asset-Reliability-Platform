from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager, redirect_stdout
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import secrets
import sys
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BenchmarkParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid benchmark arguments; use --help for bounds")

    def print_help(self, file=None):
        print(json.dumps({"help": self.format_help()}), file=file)


def bounded_integer(minimum, maximum):
    def convert(value):
        number = int(value)
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(f"expected {minimum}..{maximum}")
        return number

    return convert


def parse_args(argv=None):
    parser = BenchmarkParser(description="Local isolated ASGI/PostgreSQL FD001 benchmark")
    parser.add_argument("--assets", type=bounded_integer(1, 16), default=4)
    parser.add_argument("--cycles", type=bounded_integer(1, 362), default=60,
                        help="1..selected trajectory length (checked before connecting)")
    parser.add_argument("--concurrency", type=bounded_integer(1, 16), default=4)
    parser.add_argument("--unit-id", type=bounded_integer(1, 100), default=81)
    return parser.parse_args(argv)


def validate_workload(args, trajectory):
    if not 1 <= args.assets <= 16 or not 1 <= args.concurrency <= 16:
        raise ValueError("assets and concurrency must be between 1 and 16")
    if not 1 <= args.unit_id <= 100:
        raise ValueError("unit-id must be between 1 and 100")
    if not 1 <= args.cycles <= len(trajectory):
        raise ValueError("cycles must be between 1 and the selected trajectory length")
    if [point["cycle"] for point in trajectory] != list(range(1, len(trajectory) + 1)):
        raise ValueError("source trajectory must have contiguous ordered cycles starting at 1")


def local_database_url(value):
    if not value:
        raise ValueError("TEST_DATABASE_URL must explicitly select a local PostgreSQL database")
    try:
        url = make_url(value)
        if url.drivername not in {"postgresql", "postgresql+psycopg"}:
            raise ValueError
        if url.query or not url.username or not url.database:
            raise ValueError
        host = url.host
        if host == "localhost":
            host = "127.0.0.1"
        if not host or "%" in host or not ipaddress.ip_address(host).is_loopback:
            raise ValueError
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError
        return url.set(drivername="postgresql+psycopg", host=host, port=url.port or 5432)
    except (TypeError, ValueError):
        raise ValueError("TEST_DATABASE_URL requires a loopback PostgreSQL host, user, database and no query options") from None


def compute_metrics(latencies_ms, elapsed_seconds, errors=0):
    values = sorted(latencies_ms)
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latencies must be finite and nonnegative")
    if not isinstance(errors, int) or not 0 <= errors <= len(values):
        raise ValueError("errors must be between zero and the request count")

    def percentile(fraction):
        if not values:
            return None
        index = (len(values) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return {
        "telemetry_requests": len(values),
        "successful_requests": len(values) - errors,
        "errors": errors,
        "latency_ms": {"p50": percentile(0.50), "p95": percentile(0.95),
                       "max": max(values) if values else None},
        "percentile_method": "linear interpolation at (n-1)*p",
        "elapsed_seconds": elapsed_seconds,
        "throughput_requests_per_second": len(values) / elapsed_seconds,
        "successful_requests_per_second": (len(values) - errors) / elapsed_seconds,
    }


def validate_decisions(asset_id, predictions, alerts, orders, responses, thresholds):
    from backend.app.decision import SEVERITY_RANK

    mismatches = []
    response_errors = []
    alert = alerts[0] if len(alerts) == 1 else None
    order = orders[0] if len(orders) == 1 else None
    peak = None
    critical_seen = False
    for prediction in predictions:
        cycle = prediction.window_end_cycle
        severity = thresholds.severity(prediction.operational_rul)
        critical_seen = critical_seen or severity == "CRITICAL"
        if severity != "NORMAL" and (peak is None or SEVERITY_RANK[severity] > SEVERITY_RANK[thresholds.severity(peak.operational_rul)]):
            peak = prediction
        expected_alert = alert.id if severity != "NORMAL" and alert is not None else None
        expected_order = order.id if severity != "NORMAL" and critical_seen and order is not None else None
        if (prediction.asset_id != asset_id or not prediction.decision_processed
                or prediction.risk_severity != severity
                or prediction.decision_alert_id != expected_alert
                or prediction.decision_work_order_id != expected_order
                or (severity != "NORMAL" and alert is None)
                or (critical_seen and order is None)):
            mismatches.append(f"{asset_id}: cycle {cycle} decision linkage")
        stored = {"severity": prediction.risk_severity, "alert_id": prediction.decision_alert_id,
                  "work_order_id": prediction.decision_work_order_id}
        if cycle in responses and responses[cycle] != stored:
            response_errors.append({"asset": asset_id, "cycle": cycle, "kind": "decision_response_mismatch"})
    if len(alerts) != int(peak is not None) or (alert is not None and (
        alert.asset_id != asset_id or alert.prediction_id != peak.id
        or alert.severity != thresholds.severity(peak.operational_rul)
        or not alert.active or alert.status != "OPEN" or alert.resolved_at is not None
    )):
        mismatches.append(f"{asset_id}: alert linkage or escalation state")
    if len(orders) != int(critical_seen) or (order is not None and (
        order.asset_id != asset_id or alert is None or order.alert_id != alert.id
        or order.status != "OPEN" or order.resolved_at is not None
    )):
        mismatches.append(f"{asset_id}: work order linkage or state")
    return mismatches, response_errors


@contextmanager
def isolated_database(url, report):
    from alembic import command
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from alembic.script import ScriptDirectory
    from sqlalchemy import text
    from sqlalchemy.schema import CreateSchema, DropSchema

    from backend.app.database import build_engine

    schema = f"phase1_bench_{uuid4().hex}"
    isolated_url = url.update_query_dict({
        "hostaddr": url.host,
        "options": f"-csearch_path={schema} -cstatement_timeout=15000",
        "connect_timeout": "5",
    })
    engine = build_engine(isolated_url)
    created = False
    report["schema"] = schema
    report["cleanup"] = {"schema_created": False, "schema_dropped": False}
    try:
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
        created = True
        report["cleanup"]["schema_created"] = True
        config = Config()
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.set_main_option("sqlalchemy.url", isolated_url.render_as_string(hide_password=False).replace("%", "%%"))
        with patch.dict(os.environ):
            os.environ.pop("DATABASE_URL", None)
            command.upgrade(config, "head")
        with engine.connect() as connection:
            if connection.scalar(text("SELECT current_schema()")) != schema:
                raise RuntimeError("schema isolation failed")
            if connection.scalar(text("SHOW search_path")) != schema:
                raise RuntimeError("search path isolation failed")
            heads = list(MigrationContext.configure(connection).get_current_heads())
            if set(heads) != set(ScriptDirectory.from_config(config).get_heads()):
                raise RuntimeError("migration validation failed")
            report["alembic_heads"] = heads
            report["environment"]["postgresql"] = connection.scalar(text("SHOW server_version"))
            report["pool"] = {
                "class": type(engine.pool).__name__,
                "size": engine.pool.size(),
                "max_overflow": engine.pool._max_overflow,
                "timeout_seconds": engine.pool.timeout(),
                "recycle_seconds": engine.pool._recycle,
                "pre_ping": engine.pool._pre_ping,
                "lock_timeout": connection.scalar(text("SHOW lock_timeout")),
                "statement_timeout": connection.scalar(text("SHOW statement_timeout")),
            }
        yield engine
    finally:
        try:
            if created:
                if not schema.startswith("phase1_bench_") or len(schema) != 45:
                    raise RuntimeError("unsafe cleanup schema")
                with engine.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True))
                with engine.connect() as connection:
                    remaining = connection.scalar(
                        text("SELECT count(*) FROM pg_namespace WHERE nspname = :schema"), {"schema": schema}
                    )
                    if remaining:
                        raise RuntimeError("schema cleanup failed")
                report["cleanup"]["schema_dropped"] = True
        finally:
            engine.dispose()


async def measure(args, trajectory, engine, report):
    import anyio.to_thread
    from fastapi import FastAPI
    import httpx
    from sqlalchemy import func, select
    from sqlalchemy.orm import sessionmaker
    import torch

    from backend.app import auth, main
    from backend.app.database import get_db
    from backend.app.decision import RiskThresholds
    from backend.app.inference import RULInferenceService
    from backend.app.models import Alert, Asset, Prediction, TelemetryPointRecord, WorkOrder
    from backend.app.schemas import RULPredictionRequest

    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def benchmark_db():
        with factory() as session:
            try:
                yield session
            except BaseException:
                session.rollback()
                raise

    service = RULInferenceService()
    thresholds = RiskThresholds.from_environment()
    service.predict(RULPredictionRequest(asset_id="warmup", telemetry=trajectory[:30]))
    app = FastAPI()
    app.include_router(main.app.router)
    app.dependency_overrides[get_db] = benchmark_db
    app.dependency_overrides[main.get_inference_service] = lambda: service
    app.dependency_overrides[main.get_risk_thresholds] = lambda: thresholds
    token = secrets.token_urlsafe(32)
    credentials = (auth.Credential(hashlib.sha256(token.encode()).hexdigest(),
                                   auth.Principal("phase1-benchmark", auth.Role.ADMIN)),)
    report["auth"] = {"mode": "real_bearer_ephemeral_in_process_sha256", "bypass": False}
    report["model"] = {"id": service.manifest["model_id"], "version": service.manifest["version"],
                       "artifact_sha256": service.manifest["artifact_sha256"], "warmup_requests": 1}
    report["environment"].update(torch=str(torch.__version__), torch_threads=torch.get_num_threads(),
                                 torch_interop_threads=torch.get_num_interop_threads())
    report["settings"] = {"workers": 1, "transport": "in_process_ASGI_no_network_HTTP",
                          "assets": args.assets, "cycles_per_asset": args.cycles,
                          "concurrency": args.concurrency, "effective_concurrency": min(args.assets, args.concurrency),
                          "anyio_thread_tokens": anyio.to_thread.current_default_thread_limiter().total_tokens,
                          "risk_thresholds": {"critical": thresholds.critical, "high": thresholds.high,
                                              "medium": thresholds.medium}}
    asset_ids = [f"bench-{index}" for index in range(args.assets)]
    latencies = []
    errors = []
    decisions = {asset_id: {} for asset_id in asset_ids}
    semaphore = asyncio.Semaphore(args.concurrency)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    try:
        with patch.object(auth, "configured_credentials", return_value=credentials):
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1",
                                         trust_env=False, follow_redirects=False) as client:
                if (await client.get("/v1/auth/me")).status_code != 401:
                    raise RuntimeError("unauthenticated request was not rejected")
                client.headers["Authorization"] = f"Bearer {token}"
                principal = await client.get("/v1/auth/me")
                if principal.status_code != 200 or principal.json() != {"subject": "phase1-benchmark", "role": "admin"}:
                    raise RuntimeError("real bearer authentication failed")
                for asset_id in asset_ids:
                    response = await client.post("/v1/assets", json={"asset_id": asset_id, "name": asset_id})
                    if response.status_code != 201:
                        raise RuntimeError("asset setup failed")

                async def replay(asset_id):
                    async with semaphore:
                        for point in trajectory[:args.cycles]:
                            started = perf_counter()
                            try:
                                response = await client.post(f"/v1/assets/{asset_id}/telemetry", json=point)
                            except Exception:
                                latencies.append((perf_counter() - started) * 1000)
                                errors.append({"asset": asset_id, "cycle": point["cycle"], "kind": "request_exception"})
                                continue
                            latencies.append((perf_counter() - started) * 1000)
                            valid = response.status_code == 200
                            if valid:
                                try:
                                    body = response.json()
                                    prediction = body.get("prediction")
                                    valid = (body["created"] is True and body["cycle"] == point["cycle"]
                                             and body["asset_id"] == asset_id
                                             and ((prediction is not None) == (point["cycle"] >= 30)))
                                    if prediction is not None:
                                        valid = valid and prediction["window_end_cycle"] == point["cycle"]
                                    if valid:
                                        if point["cycle"] >= 30:
                                            decisions[asset_id][point["cycle"]] = body.get("decision")
                                        else:
                                            valid = "decision" in body and body["decision"] is None
                                except (ValueError, KeyError, TypeError):
                                    valid = False
                            if not valid:
                                errors.append({"asset": asset_id, "cycle": point["cycle"],
                                               "kind": "invalid_response", "status_code": response.status_code})

                started = perf_counter()
                async with asyncio.TaskGroup() as tasks:
                    for asset_id in asset_ids:
                        tasks.create_task(replay(asset_id))
                report["metrics"] = compute_metrics(latencies, perf_counter() - started, len(errors))
                report["request_errors"] = errors
        models = {"assets": Asset, "telemetry": TelemetryPointRecord, "predictions": Prediction,
                  "alerts": Alert, "work_orders": WorkOrder}
        mismatches = []
        expected = {"assets": args.assets, "telemetry": args.assets * args.cycles,
                    "predictions": args.assets * max(0, args.cycles - 29), "alerts": 0, "work_orders": 0}
        per_asset = {}
        with factory() as session:
            actual = {name: session.scalar(select(func.count()).select_from(model)) for name, model in models.items()}
            for asset_id in asset_ids:
                cycles = list(session.scalars(select(TelemetryPointRecord.cycle).where(
                    TelemetryPointRecord.asset_id == asset_id).order_by(TelemetryPointRecord.id)))
                predictions = list(session.scalars(select(Prediction).where(
                    Prediction.asset_id == asset_id).order_by(Prediction.window_end_cycle)))
                expected_asset = {"telemetry": args.cycles, "predictions": max(0, args.cycles - 29),
                                  "alerts": int(any(p.operational_rul <= thresholds.medium for p in predictions)),
                                  "work_orders": int(any(p.operational_rul <= thresholds.critical for p in predictions))}
                expected["alerts"] += expected_asset["alerts"]
                expected["work_orders"] += expected_asset["work_orders"]
                actual_asset = {name: session.scalar(select(func.count()).select_from(model).where(
                    model.asset_id == asset_id)) for name, model in models.items() if name != "assets"}
                per_asset[asset_id] = {"actual": actual_asset, "expected": expected_asset}
                if actual_asset != expected_asset or cycles != list(range(1, args.cycles + 1)):
                    mismatches.append(f"{asset_id}: counts or telemetry ordering")
                if [(p.window_start_cycle, p.window_end_cycle) for p in predictions] != [
                    (cycle - 29, cycle) for cycle in range(30, args.cycles + 1)
                ] or not all(p.decision_processed and p.model_version == service.manifest["version"]
                             and p.model_id == service.manifest["model_id"] and math.isfinite(p.raw_rul)
                             and p.operational_rul == max(p.raw_rul, 0.0)
                             and p.risk_severity == thresholds.severity(p.operational_rul) for p in predictions):
                    mismatches.append(f"{asset_id}: prediction validation")
                alerts = list(session.scalars(select(Alert).where(Alert.asset_id == asset_id)))
                orders = list(session.scalars(select(WorkOrder).where(WorkOrder.asset_id == asset_id)))
                decision_mismatches, response_errors = validate_decisions(
                    asset_id, predictions, alerts, orders, decisions[asset_id], thresholds
                )
                mismatches.extend(decision_mismatches)
                errors.extend(response_errors)
        report["metrics"] = compute_metrics(latencies, report["metrics"]["elapsed_seconds"], len(errors))
        if actual != expected or len(latencies) != args.assets * args.cycles:
            mismatches.append("total counts")
        report["persisted"] = {"actual": actual, "expected": expected, "per_asset": per_asset,
                               "workflow_expectation": "derived from persisted real-model RUL and configured thresholds"}
        report["validation"] = {"passed": not errors and not mismatches, "mismatches": mismatches}
        report["status"] = "passed" if report["validation"]["passed"] else "failed"
    finally:
        app.dependency_overrides.clear()


def run(args, report=None):
    report = report if report is not None else {}
    report.update(status="failed", environment={"python": platform.python_version(), "platform": platform.platform()})
    url = local_database_url(os.getenv("TEST_DATABASE_URL"))
    with patch.dict(os.environ):
        for name in list(os.environ):
            if name.startswith("PG") and name != "PG_LOCK_TIMEOUT_MS":
                os.environ.pop(name)
        from backend.app.replay import load_training_trajectory

        trajectory = load_training_trajectory(ROOT / "data" / "raw" / "source" / "CMAPSSData", args.unit_id)
        validate_workload(args, trajectory)
        report["source"] = {"dataset": "FD001", "split": "train", "unit_id": args.unit_id,
                            "trajectory_cycles": len(trajectory), "first_cycle": 1, "last_cycle": args.cycles}
        with isolated_database(url, report) as engine:
            asyncio.run(measure(args, trajectory, engine, report))
    return report


def main(argv=None):
    report = {}
    started = perf_counter()
    try:
        args = parse_args(argv)
        with redirect_stdout(sys.stderr):
            run(args, report)
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": "benchmark failed; connection details suppressed"}
    report["total_elapsed_seconds"] = perf_counter() - started
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
