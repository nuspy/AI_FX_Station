"""add regime_label to latents

Revision ID: 0007_add_regime_label
Revises: 0006_add_ticks_table
Create Date: 2025-09-07 02:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0007_add_regime_label"
down_revision = "0006_add_ticks_table"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("latents", sa.Column("regime_label", sa.String(64), nullable=True))
    op.create_index("ix_latents_regime_label", "latents", ["regime_label"], unique=False)


def downgrade():
    op.drop_index("ix_latents_regime_label", table_name="latents")
    op.drop_column("latents", "regime_label")
