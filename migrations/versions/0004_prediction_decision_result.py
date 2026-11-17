from alembic import op
import sqlalchemy as sa


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None
sqlite_batch_recreate = True


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.add_column(
            sa.Column("decision_processed", sa.Boolean(), server_default=sa.false(), nullable=False)
        )
        batch.add_column(sa.Column("decision_alert_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("decision_work_order_id", sa.Integer(), nullable=True))
        batch.create_foreign_key(
            "fk_prediction_decision_alert", "alerts", ["decision_alert_id"], ["id"]
        )
        batch.create_foreign_key(
            "fk_prediction_decision_work_order", "work_orders", ["decision_work_order_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.drop_constraint("fk_prediction_decision_work_order", type_="foreignkey")
        batch.drop_constraint("fk_prediction_decision_alert", type_="foreignkey")
        batch.drop_column("decision_work_order_id")
        batch.drop_column("decision_alert_id")
        batch.drop_column("decision_processed")
