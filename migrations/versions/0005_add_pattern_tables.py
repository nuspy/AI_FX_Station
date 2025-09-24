"""add pattern tables"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0005_add_pattern_tables'
down_revision = '0004_add_backtesting_tables'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('pattern_defs',
        sa.Column('key', sa.String(64), primary_key=True),
        sa.Column('family', sa.String(64), nullable=True),
        sa.Column('kind', sa.String(16), nullable=False), # chart|candle
        sa.Column('name', sa.String(128), nullable=False),
        sa.Column('effect', sa.String(32), nullable=True),
        sa.Column('dsl_json', sa.Text(), nullable=True),
        sa.Column('image_res', sa.String(256), nullable=True),
        sa.Column('links_json', sa.Text(), nullable=True),
    )
    op.create_table('pattern_benchmarks',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('key', sa.String(64), sa.ForeignKey('pattern_defs.key'), nullable=False),
        sa.Column('performance_rank', sa.String(64)),
        sa.Column('breakeven_failure_rate', sa.String(64)),
        sa.Column('average_rise', sa.String(64)),
        sa.Column('change_after_trend_ends', sa.String(64)),
        sa.Column('volume_trend', sa.String(64)),
        sa.Column('throwbacks', sa.String(64)),
        sa.Column('pct_meet_target', sa.String(64)),
        sa.Column('surprising_findings', sa.Text()),
        sa.Column('bull_notes', sa.Text()),
        sa.Column('bear_notes', sa.Text()),
    )
    op.create_table('pattern_events',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('key', sa.String(64), sa.ForeignKey('pattern_defs.key'), nullable=False),
        sa.Column('symbol', sa.String(32), nullable=False),
        sa.Column('timeframe', sa.String(16), nullable=False),
        sa.Column('start_ts', sa.BigInteger, nullable=False),
        sa.Column('confirm_ts', sa.BigInteger, nullable=False),
        sa.Column('direction', sa.String(8), nullable=True),
        sa.Column('kind', sa.String(16), nullable=False),
        sa.Column('state', sa.String(16), nullable=False),
        sa.Column('score', sa.Float, nullable=True),
        sa.Column('scale_atr', sa.Float, nullable=True),
        sa.Column('bars_span', sa.Integer, nullable=True),
        sa.Column('target_price', sa.Float, nullable=True),
        sa.Column('horizon_bars', sa.Integer, nullable=True),
        sa.Column('overlay_json', sa.Text(), nullable=True),
    )

def downgrade():
    op.drop_table('pattern_events')
    op.drop_table('pattern_benchmarks')
    op.drop_table('pattern_defs')
