"""Create assets, telemetry points, and predictions."""

from alembic import op
import sqlalchemy as sa


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assets",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    telemetry_columns = [
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("asset_id", sa.String(length=128), sa.ForeignKey("assets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("cycle", sa.Integer(), nullable=False),
        sa.Column("operational_setting_1", sa.Float(), nullable=False),
        sa.Column("operational_setting_2", sa.Float(), nullable=False),
        sa.Column("operational_setting_3", sa.Float(), nullable=False),
        *[sa.Column(f"sensor_{index}", sa.Float(), nullable=False) for index in range(1, 22)],
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("asset_id", "cycle", name="uq_telemetry_asset_cycle"),
    ]
    op.create_table("telemetry_points", *telemetry_columns)
    op.create_index("ix_telemetry_points_asset_id", "telemetry_points", ["asset_id"])
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("asset_id", sa.String(length=128), sa.ForeignKey("assets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("window_start_cycle", sa.Integer(), nullable=False),
        sa.Column("window_end_cycle", sa.Integer(), nullable=False),
        sa.Column("raw_rul", sa.Float(), nullable=False),
        sa.Column("operational_rul", sa.Float(), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("asset_id", "window_end_cycle", "model_version", name="uq_prediction_asset_cycle_model"),
    )
    op.create_index("ix_predictions_asset_id", "predictions", ["asset_id"])


def downgrade() -> None:
    op.drop_index("ix_predictions_asset_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_telemetry_points_asset_id", table_name="telemetry_points")
    op.drop_table("telemetry_points")
    op.drop_table("assets")

