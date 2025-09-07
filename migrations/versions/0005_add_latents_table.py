"""add latents table

Revision ID: 0005_add_latents_table
Revises: 0004_features_gin_jsonb_index
Create Date: 2025-09-07 01:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0005_add_latents_table"
down_revision = "0004_features_gin_jsonb_index"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "latents",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(64), nullable=True, index=True),
        sa.Column("timeframe", sa.String(16), nullable=True, index=True),
        sa.Column("ts_utc", sa.Integer, nullable=False, index=True),
        sa.Column("model_version", sa.String(128), nullable=True),
        sa.Column("latent_json", sa.Text, nullable=False),
        sa.Column("ts_created_ms", sa.Integer, nullable=False),
    )
    op.create_index("ix_latents_symbol_tf_ts", "latents", ["symbol", "timeframe", "ts_utc"], unique=False)


def downgrade():
    op.drop_index("ix_latents_symbol_tf_ts", table_name="latents")
    op.drop_table("latents")
