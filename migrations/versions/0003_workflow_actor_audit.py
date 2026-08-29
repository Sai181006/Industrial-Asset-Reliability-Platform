"""Record the authenticated actor on workflow mutations."""

from alembic import op
import sqlalchemy as sa


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("alerts", sa.Column("last_updated_by", sa.String(length=128), nullable=True))
    op.add_column("work_orders", sa.Column("last_updated_by", sa.String(length=128), nullable=True))


def downgrade() -> None:
    op.drop_column("work_orders", "last_updated_by")
    op.drop_column("alerts", "last_updated_by")
