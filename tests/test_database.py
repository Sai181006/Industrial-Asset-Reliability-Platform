import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from alembic import command
from alembic.config import Config
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.pool import SingletonThreadPool

from backend.app import database
from backend.app.decision import RiskThresholds, apply_decision, stored_decision
from backend.app.main import app, get_inference_service
from backend.app.models import Alert, Asset, Prediction, TelemetryPointRecord, WorkOrder
from ml.baseline import FEATURE_COLUMNS


def session_for(backend="postgresql"):
    session = Mock()
    session.get_bind.return_value.dialect.name = backend
    return session


def transaction_error(code, invalidated=False):
    original = Exception("database failure")
    original.sqlstate = code
    return OperationalError("statement", {}, original, connection_invalidated=invalidated)


@pytest.mark.parametrize("code", ["40001", "40P01", "55P03"])
@pytest.mark.parametrize("backend", ["postgresql", "sqlite"])
def test_retryable_errors_are_postgresql_only(code, backend):
    assert database.is_retryable_transaction_error(transaction_error(code), session_for(backend)) == (
        backend == "postgresql"
    )


@pytest.mark.parametrize("code", [None, "08006", "08007", "23505", "57014"])
def test_other_database_errors_are_not_retryable(code):
    assert not database.is_retryable_transaction_error(transaction_error(code), session_for())


def test_invalidated_connection_is_not_retryable():
    assert not database.is_retryable_transaction_error(transaction_error("40001", True), session_for())


@pytest.mark.parametrize("failure_at", ["operation", "commit"])
def test_atomic_mutation_retries_the_whole_transaction(monkeypatch, failure_at):
    session = session_for()
    events = []
    session.rollback.side_effect = lambda: events.append("rollback")
    monkeypatch.setattr(database, "sleep", lambda delay: events.append("sleep"))
    operation_failure = [True]
    commit_failure = [True]

    def commit():
        events.append("commit")
        if failure_at == "commit" and commit_failure.pop():
            commit_failure.append(False)
            raise transaction_error("40001")

    session.commit.side_effect = commit

    @database.atomic_mutation
    def operation(db):
        events.append("operation")
        if failure_at == "operation" and operation_failure.pop():
            operation_failure.append(False)
            raise transaction_error("40P01")
        return "result"

    assert operation(db=session) == "result"
    assert events == (
        ["operation", "rollback", "sleep", "operation", "commit"]
        if failure_at == "operation"
        else ["operation", "commit", "rollback", "sleep", "operation", "commit"]
    )


def test_atomic_mutation_stops_after_three_attempts(monkeypatch):
    session = session_for()
    monkeypatch.setattr(database, "sleep", Mock())
    calls = Mock(side_effect=transaction_error("55P03"))

    @database.atomic_mutation
    def operation(db):
        return calls()

    with pytest.raises(OperationalError):
        operation(session)
    assert calls.call_count == 3
    assert session.rollback.call_count == 3
    session.commit.assert_not_called()
    assert database.sleep.call_count == 2


@pytest.mark.parametrize("failure", [ValueError("inference failed"), transaction_error("08007"), transaction_error("40001", True)])
def test_atomic_mutation_rolls_back_without_unsafe_retries(monkeypatch, failure):
    session = session_for()
    sleeper = Mock()
    monkeypatch.setattr(database, "sleep", sleeper)
    calls = Mock(side_effect=failure)

    @database.atomic_mutation
    def operation(db):
        return calls()

    with pytest.raises(type(failure)):
        operation(session)
    calls.assert_called_once()
    session.rollback.assert_called_once()
    session.commit.assert_not_called()
    sleeper.assert_not_called()


@pytest.mark.parametrize("code,constraint,expected", [
    ("23505", "assets_pkey", True),
    ("23505", "other_key", False),
    ("23503", "assets_pkey", False),
    (None, "assets_pkey", False),
])
def test_postgresql_duplicate_asset_requires_exact_constraint(code, constraint, expected):
    original = Exception("integrity failure")
    original.sqlstate = code
    original.diag = SimpleNamespace(constraint_name=constraint)
    error = IntegrityError("insert", {}, original)
    assert database.is_duplicate_asset(error, session_for()) is expected


def test_sqlite_duplicate_asset_is_recognized():
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE assets (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
        connection.execute("INSERT INTO assets VALUES ('asset', 'name')")
        with pytest.raises(sqlite3.IntegrityError) as duplicate:
            connection.execute("INSERT INTO assets VALUES ('asset', 'name')")
        assert database.is_duplicate_asset(IntegrityError("insert", {}, duplicate.value), session_for("sqlite"))
        with pytest.raises(sqlite3.IntegrityError) as not_null:
            connection.execute("INSERT INTO assets VALUES ('other', NULL)")
        assert not database.is_duplicate_asset(IntegrityError("insert", {}, not_null.value), session_for("sqlite"))


@pytest.mark.parametrize("name,value", [
    ("DB_POOL_SIZE", "0"),
    ("DB_POOL_SIZE", "1001"),
    ("DB_MAX_OVERFLOW", "-1"),
    ("DB_POOL_TIMEOUT", "0"),
    ("DB_POOL_TIMEOUT", "nan"),
    ("DB_POOL_RECYCLE", "-2"),
    ("PG_LOCK_TIMEOUT_MS", "0"),
    ("PG_LOCK_TIMEOUT_MS", "2147483648"),
    ("PG_LOCK_TIMEOUT_MS", "1; SELECT 1"),
])
def test_engine_settings_are_validated(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        database.build_engine("sqlite://")


def test_engine_pool_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_POOL_SIZE", "2")
    monkeypatch.setenv("DB_MAX_OVERFLOW", "3")
    monkeypatch.setenv("DB_POOL_TIMEOUT", "4")
    monkeypatch.setenv("DB_POOL_RECYCLE", "5")
    engine = database.build_engine(f"sqlite:///{(tmp_path / 'pool.db').as_posix()}")
    try:
        assert engine.pool.size() == 2
        assert engine.pool._max_overflow == 3
        assert engine.pool.timeout() == 4
        assert engine.pool._recycle == 5
    finally:
        engine.dispose()


def test_memory_sqlite_keeps_compatible_pool():
    engine = database.build_engine("sqlite://")
    try:
        assert isinstance(engine.pool, SingletonThreadPool)
    finally:
        engine.dispose()


def test_postgresql_lock_timeout_is_transaction_local(monkeypatch):
    configured_engine = Mock()
    listener = Mock(side_effect=lambda function: function)
    monkeypatch.setattr(database, "create_engine", Mock(return_value=configured_engine))
    monkeypatch.setattr(database.event, "listens_for", Mock(return_value=listener))
    monkeypatch.setenv("PG_LOCK_TIMEOUT_MS", "1234")
    assert database.build_engine("postgresql+psycopg://localhost/test") is configured_engine
    database.event.listens_for.assert_called_once_with(configured_engine, "begin")
    connection = Mock()
    listener.call_args.args[0](connection)
    connection.exec_driver_sql.assert_called_once_with("SET LOCAL lock_timeout = '1234ms'")


def test_asset_lock_compiles_for_postgresql_and_is_ignored_by_sqlite():
    session = session_for()
    database.lock_asset(session, "asset")
    statement = session.scalar.call_args.args[0]
    assert str(statement.compile(dialect=postgresql.dialect())).endswith("FOR UPDATE")
    assert "FOR UPDATE" not in str(statement.compile(dialect=sqlite.dialect()))
    assert statement.get_execution_options()["populate_existing"] is True
    assert statement.column_descriptions[0]["entity"] is Asset


def test_get_db_rolls_back_and_closes_on_error(monkeypatch):
    session = session_for()
    monkeypatch.setattr(database, "SessionLocal", Mock(return_value=session))
    dependency = database.get_db()
    assert next(dependency) is session
    with pytest.raises(ValueError):
        dependency.throw(ValueError("request failed"))
    session.rollback.assert_called_once()
    session.close.assert_called_once()


@pytest.fixture()
def decision_session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'decisions.db').as_posix()}")
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with factory() as session:
        session.add(Asset(id="asset", name="Asset"))
        session.commit()
    try:
        yield factory
    finally:
        engine.dispose()


def new_prediction(cycle=30, rul=5):
    return Prediction(
        asset_id="asset",
        window_start_cycle=cycle - 29,
        window_end_cycle=cycle,
        raw_rul=rul,
        operational_rul=rul,
        model_id="model",
        model_version="v1",
    )


def test_stored_legacy_decision_rejects_without_database_calls():
    session = session_for()
    prediction = new_prediction()
    prediction.decision_processed = False
    with pytest.raises(HTTPException) as error:
        stored_decision(session, prediction)
    assert error.value.status_code == 409
    assert error.value.detail == "legacy prediction requires decision reconciliation"
    assert prediction.decision_processed is False
    assert prediction.decision_alert_id is None
    assert prediction.decision_work_order_id is None
    assert session.method_calls == []


@pytest.mark.parametrize("alert_state", [None, "OPEN", "RESOLVED", "SHARED"])
def test_legacy_ingestion_retry_blocks_without_inference_or_writes(decision_session_factory, alert_state):
    points = [{"cycle": cycle, **dict.fromkeys(FEATURE_COLUMNS, 0.0)} for cycle in range(1, 31)]
    with decision_session_factory() as session:
        prediction = new_prediction()
        session.add(prediction)
        session.add_all(TelemetryPointRecord(asset_id="asset", **point) for point in points)
        session.flush()
        if alert_state is not None:
            source = prediction
            if alert_state == "SHARED":
                source = new_prediction(cycle=31)
                session.add(source)
                session.flush()
            alert = Alert(
                asset_id="asset", prediction_id=source.id, severity="CRITICAL",
                status="RESOLVED" if alert_state == "RESOLVED" else "OPEN",
                active=alert_state != "RESOLVED",
            )
            session.add(alert)
            session.flush()
            session.add(WorkOrder(asset_id="asset", alert_id=alert.id, status=alert.status))
        session.commit()
        prediction_id = prediction.id
        before = {
            model: session.execute(select(model.__table__)).all()
            for model in (Prediction, Alert, WorkOrder, TelemetryPointRecord)
        }
    service = Mock(manifest={"version": "v1"})

    def override_get_db():
        with decision_session_factory() as session:
            yield session

    app.dependency_overrides[database.get_db] = override_get_db
    app.dependency_overrides[get_inference_service] = lambda: service
    try:
        with TestClient(app) as client:
            for _ in range(2):
                response = client.post("/v1/assets/asset/telemetry", json=points[-1])
                assert response.status_code == 409
                assert response.json() == {"detail": "legacy prediction requires decision reconciliation"}
        service.predict.assert_not_called()
        with decision_session_factory() as session:
            assert session.get(Prediction, prediction_id).decision_processed is False
            for model, rows in before.items():
                assert session.execute(select(model.__table__)).all() == rows
    finally:
        app.dependency_overrides.pop(database.get_db, None)
        app.dependency_overrides.pop(get_inference_service, None)


@pytest.mark.parametrize("processed", [False, True])
def test_stale_prediction_refreshes_after_lock_without_reopening(decision_session_factory, monkeypatch, processed):
    with decision_session_factory() as stale:
        prediction = new_prediction()
        prediction.decision_processed = processed
        stale.add(prediction)
        stale.commit()
        with decision_session_factory() as current:
            fresh = current.get(Prediction, prediction.id)
            fresh.decision_processed = False
            result = apply_decision(current, fresh, RiskThresholds())
            alert = current.get(Alert, result.alert_id)
            order = current.get(WorkOrder, result.work_order_id)
            alert.status = "RESOLVED"
            alert.active = False
            order.status = "RESOLVED"
            current.commit()
        assert prediction.decision_processed is processed
        assert prediction.decision_alert_id is None
        events = []
        from backend.app import decision

        original_lock = decision.lock_asset
        original_refresh = stale.refresh

        def lock(db, asset_id):
            events.append("lock")
            return original_lock(db, asset_id)

        def refresh(instance):
            events.append("refresh")
            return original_refresh(instance)

        monkeypatch.setattr(decision, "lock_asset", lock)
        monkeypatch.setattr(stale, "refresh", refresh)
        assert apply_decision(stale, prediction, RiskThresholds()) == result
        assert events == ["lock", "refresh"]
        assert prediction.decision_processed is True
        assert prediction.risk_severity == "CRITICAL"
        stale.commit()
    with decision_session_factory() as session:
        alerts = session.scalars(select(Alert)).all()
        orders = session.scalars(select(WorkOrder)).all()
        assert len(alerts) == len(orders) == 1
        assert alerts[0].status == orders[0].status == "RESOLVED"
        assert alerts[0].active is False


@pytest.mark.parametrize("pending", [False, True])
def test_decision_refresh_preserves_new_prediction_and_required_changes(decision_session_factory, pending):
    with decision_session_factory() as session:
        prediction = new_prediction(rul=100)
        session.add(prediction)
        if not pending:
            session.flush()
        prediction.operational_rul = 5
        result = apply_decision(session, prediction, RiskThresholds())
        session.commit()
        session.refresh(prediction)
        assert result.severity == "CRITICAL"
        assert prediction.operational_rul == 5
        assert prediction.decision_processed is True
        assert prediction.decision_alert_id == result.alert_id
        assert prediction.decision_work_order_id == result.work_order_id
        assert result.alert_id is not None
        assert result.work_order_id is not None


def test_decision_migration_preserves_legacy_predictions(monkeypatch, tmp_path):
    url = f"sqlite:///{(tmp_path / 'migration.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    config = Config(str(database.ROOT / "alembic.ini"))
    command.upgrade(config, "0003")
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("INSERT INTO assets (id, name) VALUES ('asset', 'Asset')")
            connection.exec_driver_sql(
                "INSERT INTO predictions (id, asset_id, window_start_cycle, window_end_cycle, "
                "raw_rul, operational_rul, model_id, model_version, risk_severity) "
                "VALUES (1, 'asset', 1, 30, 5, 5, 'model', 'v1', 'CRITICAL')"
            )
            connection.exec_driver_sql(
                "INSERT INTO alerts (id, asset_id, prediction_id, severity, status, active) "
                "VALUES (1, 'asset', 1, 'CRITICAL', 'RESOLVED', 0)"
            )
        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT decision_processed, decision_alert_id, decision_work_order_id, risk_severity "
                "FROM predictions WHERE id = 1"
            ).one()
            assert tuple(row) == (0, None, None, "CRITICAL")
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        foreign_keys = inspect(engine).get_foreign_keys("predictions")
        assert {key["name"] for key in foreign_keys} >= {
            "fk_prediction_decision_alert", "fk_prediction_decision_work_order"
        }
        command.downgrade(config, "0003")
        assert "decision_processed" not in {column["name"] for column in inspect(engine).get_columns("predictions")}
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT risk_severity FROM predictions WHERE id = 1").scalar_one() == "CRITICAL"
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    finally:
        engine.dispose()
