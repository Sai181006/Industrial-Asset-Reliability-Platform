from alembic import op
import sqlalchemy as sa


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None
sqlite_batch_recreate = True


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.add_column(sa.Column("drivers", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.drop_column("drivers")
