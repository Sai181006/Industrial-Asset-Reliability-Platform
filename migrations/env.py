from logging.config import fileConfig
import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.app.database import Base
from backend.app import models  # noqa: F401


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
if os.getenv("DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    if connection.dialect.name != "sqlite":
        with context.begin_transaction():
            context.run_migrations()
        return

    caller_transaction = connection.in_transaction() or connection.connection.driver_connection.in_transaction
    migrations_fn = context.get_context().opts["fn"]
    foreign_keys = None

    def restore_foreign_keys():
        nonlocal foreign_keys
        if foreign_keys is not None:
            connection.rollback()
            cursor = connection.connection.cursor()
            try:
                cursor.execute(f"PRAGMA foreign_keys = {foreign_keys}")
                if cursor.execute("PRAGMA foreign_keys").fetchone()[0] != foreign_keys:
                    raise RuntimeError("Could not restore SQLite foreign key enforcement")
            finally:
                cursor.close()
                foreign_keys = None

    def migration_steps(heads, migration_context):
        nonlocal foreign_keys
        for step in migrations_fn(heads, migration_context):
            if getattr(step.revision.module, "sqlite_batch_recreate", False):
                if caller_transaction:
                    raise RuntimeError("SQLite batch migration requires a connection without an active transaction")
                connection.commit()
                cursor = connection.connection.cursor()
                try:
                    foreign_keys = cursor.execute("PRAGMA foreign_keys").fetchone()[0]
                    cursor.execute("PRAGMA foreign_keys = OFF")
                    if cursor.execute("PRAGMA foreign_keys").fetchone()[0] != 0:
                        raise RuntimeError("Could not disable SQLite foreign key enforcement")
                finally:
                    cursor.close()
                connection.begin()
                if not connection.connection.driver_connection.in_transaction:
                    connection.exec_driver_sql("BEGIN")
            yield step
            restore_foreign_keys()

    def check_foreign_keys(**kwargs):
        if foreign_keys is not None:
            violations = connection.exec_driver_sql("PRAGMA foreign_key_check").all()
            if violations:
                raise RuntimeError(f"SQLite foreign_key_check failed: {violations}")

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        fn=migration_steps,
        on_version_apply=check_foreign_keys,
    )
    try:
        with context.begin_transaction():
            context.run_migrations()
    finally:
        restore_foreign_keys()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        run_migrations(connection)
        return
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        with connectable.connect() as connection:
            run_migrations(connection)
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

