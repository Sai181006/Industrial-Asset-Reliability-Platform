from io import StringIO
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def migration_connection(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    engine = create_engine(f"sqlite:///{(tmp_path / 'migration.db').as_posix()}")
    config = Config()
    config.set_main_option("script_location", str(ROOT / "migrations"))
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys = ON")
            connection.commit()
            config.attributes["connection"] = connection
            command.upgrade(config, "0003")
            connection.exec_driver_sql("INSERT INTO assets (id, name) VALUES ('asset', 'Asset')")
            connection.exec_driver_sql(
                "INSERT INTO predictions (id, asset_id, window_start_cycle, window_end_cycle, "
                "raw_rul, operational_rul, model_id, model_version, risk_severity) "
                "VALUES (1, 'asset', 1, 30, 5, 5, 'model', 'v1', 'CRITICAL'), "
                "(2, 'asset', 2, 31, 25, 25, 'model', 'v1', 'HIGH')"
            )
            connection.exec_driver_sql(
                "INSERT INTO alerts (id, asset_id, prediction_id, severity, status, active, last_updated_by) "
                "VALUES (11, 'asset', 1, 'CRITICAL', 'RESOLVED', 0, 'engineer'), "
                "(12, 'asset', 2, 'HIGH', 'OPEN', 1, 'engineer')"
            )
            connection.exec_driver_sql(
                "INSERT INTO work_orders (id, asset_id, alert_id, status, notes, last_updated_by) "
                "VALUES (21, 'asset', 11, 'RESOLVED', 'repaired', 'engineer'), "
                "(22, 'asset', 12, 'OPEN', 'inspect', 'engineer')"
            )
            connection.commit()
            yield config, connection
    finally:
        engine.dispose()


def snapshot(connection):
    return {
        table: connection.exec_driver_sql(f"SELECT * FROM {table} ORDER BY id").all()
        for table in ("assets", "predictions", "alerts", "work_orders")
    }


@pytest.mark.parametrize("enabled", [0, 1])
def test_populated_upgrade_and_downgrade_preserve_references(migration_connection, enabled):
    config, connection = migration_connection
    connection.exec_driver_sql(f"PRAGMA foreign_keys = {enabled}")
    before = snapshot(connection)
    connection.commit()
    command.upgrade(config, "0004")
    assert not connection.closed
    assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == enabled
    assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    assert connection.exec_driver_sql(
        "SELECT decision_processed, decision_alert_id, decision_work_order_id FROM predictions ORDER BY id"
    ).all() == [(0, None, None), (0, None, None)]
    keys = inspect(connection).get_foreign_keys("predictions")
    assert {key["name"] for key in keys} >= {
        "fk_prediction_decision_alert", "fk_prediction_decision_work_order"
    }
    after = snapshot(connection)
    assert [tuple(row[:len(before['predictions'][0])]) for row in after["predictions"]] == before["predictions"]
    for table in ("assets", "alerts", "work_orders"):
        assert after[table] == before[table]
    connection.exec_driver_sql(
        "UPDATE predictions SET decision_processed = 1, decision_alert_id = 11, decision_work_order_id = 21"
    )
    connection.commit()
    command.downgrade(config, "0003")
    assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == enabled
    assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    assert snapshot(connection) == before
    assert "decision_processed" not in {column["name"] for column in inspect(connection).get_columns("predictions")}
    if enabled:
        with pytest.raises(IntegrityError):
            connection.exec_driver_sql("DELETE FROM predictions WHERE id = 1")
        connection.rollback()


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
@pytest.mark.parametrize("enabled", [0, 1])
@pytest.mark.parametrize("failure", ["batch", "foreign_key_check"])
def test_failure_restores_enforcement_and_rolls_back(migration_connection, direction, enabled, failure):
    config, connection = migration_connection
    if direction == "downgrade":
        command.upgrade(config, "0004")
    connection.exec_driver_sql(f"PRAGMA foreign_keys = {enabled}")
    before = snapshot(connection)
    connection.commit()

    def inject_failure(conn, cursor, statement, parameters, execution_context, executemany):
        if statement.strip() == "DROP TABLE predictions":
            assert conn.connection.driver_connection.in_transaction
            assert cursor.execute("PRAGMA foreign_keys").fetchone()[0] == 0
            if failure == "batch":
                raise RuntimeError("injected batch failure")
            cursor.execute("UPDATE alerts SET prediction_id = 999 WHERE id = 11")

    event.listen(connection, "before_cursor_execute", inject_failure)
    try:
        with pytest.raises(RuntimeError, match="injected batch failure|foreign_key_check"):
            if direction == "upgrade":
                command.upgrade(config, "0004")
            else:
                command.downgrade(config, "0003")
    finally:
        event.remove(connection, "before_cursor_execute", inject_failure)
    assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == enabled
    assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    assert snapshot(connection) == before
    assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == (
        "0003" if direction == "upgrade" else "0004"
    )
    assert "_alembic_tmp_predictions" not in inspect(connection).get_table_names()


def test_batch_rejects_caller_transaction_without_committing(migration_connection):
    config, connection = migration_connection
    connection.exec_driver_sql("UPDATE assets SET name = 'pending' WHERE id = 'asset'")
    with pytest.raises(RuntimeError, match="active transaction"):
        command.upgrade(config, "0004")
    assert connection.in_transaction()
    assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
    connection.rollback()
    assert connection.exec_driver_sql("SELECT name FROM assets").scalar_one() == "Asset"
