"""
Add backtesting tables: bt_job, bt_config, bt_result, bt_trace

Revision ID: 0004_add_backtesting_tables
Revises: 0003_fix_ticks_and_constraints
Create Date: 2025-09-17 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0004_add_backtesting_tables"
down_revision = "0003_fix_ticks_and_constraints"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "bt_job",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=True),
        sa.Column("market_calendar", sa.String(length=64), nullable=True),
    )

    op.create_table(
        "bt_config",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.Integer(), nullable=True, index=True),
        sa.Column("fingerprint", sa.String(length=64), nullable=False, unique=True),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("dropped", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("drop_reason", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.Integer(), nullable=True),
        sa.Column("ended_at", sa.Integer(), nullable=True),
        sa.Column("horizons_raw", sa.Text(), nullable=True),
        sa.Column("horizons_sec", sa.JSON(), nullable=True),
        sa.Column("start_ts", sa.Integer(), nullable=True),
        sa.Column("end_ts", sa.Integer(), nullable=True),
        sa.Column("interval_type", sa.String(length=16), nullable=True),
        sa.Column("time_filter_json", sa.JSON(), nullable=True),
        sa.Column("walkforward_json", sa.JSON(), nullable=True),
    )

    op.create_table(
        "bt_result",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("config_id", sa.Integer(), nullable=False, index=True),
        sa.Column("adherence_mean", sa.Float(), nullable=True),
        sa.Column("adherence_std", sa.Float(), nullable=True),
        sa.Column("p10", sa.Float(), nullable=True),
        sa.Column("p25", sa.Float(), nullable=True),
        sa.Column("p50", sa.Float(), nullable=True),
        sa.Column("p75", sa.Float(), nullable=True),
        sa.Column("p90", sa.Float(), nullable=True),
        sa.Column("win_rate_delta", sa.Float(), nullable=True),
        sa.Column("delta_used", sa.Float(), nullable=True),
        sa.Column("skill_rw", sa.Float(), nullable=True),
        sa.Column("coverage_observed", sa.Float(), nullable=True),
        sa.Column("coverage_target", sa.Float(), nullable=True),
        sa.Column("coverage_abs_error", sa.Float(), nullable=True),
        sa.Column("band_efficiency", sa.Float(), nullable=True),
        sa.Column("n_points", sa.Integer(), nullable=True),
        sa.Column("cv_horizons", sa.Float(), nullable=True),
        sa.Column("cv_time", sa.Float(), nullable=True),
        sa.Column("robustness_index", sa.Float(), nullable=True),
        sa.Column("complexity_penalty", sa.Float(), nullable=True),
        sa.Column("composite_score", sa.Float(), nullable=True),
        sa.Column("horizon_profile_json", sa.JSON(), nullable=True),
        sa.Column("time_profile_json", sa.JSON(), nullable=True),
        sa.Column("coverage_ratio", sa.Float(), nullable=True),
    )

    op.create_table(
        "bt_trace",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("config_id", sa.Integer(), nullable=False, index=True),
        sa.Column("slice_idx", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
    )

    # Indexes
    op.create_index("ix_bt_cfg_fingerprint", "bt_config", ["fingerprint"], unique=True)
    op.create_index("ix_bt_cfg_job", "bt_config", ["job_id"], unique=False)
    op.create_index("ix_bt_res_cfg", "bt_result", ["config_id"], unique=False)
    op.create_index("ix_bt_trace_cfg_slice", "bt_trace", ["config_id", "slice_idx"], unique=False)


def downgrade():
    op.drop_index("ix_bt_trace_cfg_slice", table_name="bt_trace")
    op.drop_index("ix_bt_res_cfg", table_name="bt_result")
    op.drop_index("ix_bt_cfg_job", table_name="bt_config")
    op.drop_index("ix_bt_cfg_fingerprint", table_name="bt_config")
    op.drop_table("bt_trace")
    op.drop_table("bt_result")
    op.drop_table("bt_config")
    op.drop_table("bt_job")


