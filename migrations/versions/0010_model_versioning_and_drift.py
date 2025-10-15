"""Model versioning and drift tracking

Revision ID: 0010
Revises: 0009
Create Date: 2025-10-07

Adds model versioning, drift detection, and production status tracking
for automated model lifecycle management.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic
revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade database schema."""
    # Add versioning to models table (assuming it exists)
    # If models table doesn't exist, create it
    op.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW(),
            metadata JSONB
        )
    """)

    # Add version tracking
    op.add_column('models',
                  sa.Column('model_version',
                           sa.String(20),
                           nullable=True))

    # Add CV strategy metadata
    op.add_column('models',
                  sa.Column('training_cv_strategy',
                           sa.String(20),
                           nullable=True))

    # Add drift score
    op.add_column('models',
                  sa.Column('drift_score',
                           sa.Float,
                           nullable=True))

    # Add production status
    op.add_column('models',
                  sa.Column('is_production',
                           sa.Boolean,
                           nullable=False,
                           server_default='false'))

    # Add deployment timestamp
    op.add_column('models',
                  sa.Column('deployed_at',
                           sa.TIMESTAMP,
                           nullable=True))

    # Add performance metrics
    op.add_column('models',
                  sa.Column('validation_rmse',
                           sa.Float,
                           nullable=True))

    op.add_column('models',
                  sa.Column('validation_sharpe',
                           sa.Float,
                           nullable=True))

    # Create index for production models
    op.create_index('idx_models_production',
                   'models',
                   ['is_production'])


def downgrade():
    """Downgrade database schema."""
    op.drop_index('idx_models_production', table_name='models')
    op.drop_column('models', 'validation_sharpe')
    op.drop_column('models', 'validation_rmse')
    op.drop_column('models', 'deployed_at')
    op.drop_column('models', 'is_production')
    op.drop_column('models', 'drift_score')
    op.drop_column('models', 'training_cv_strategy')
    op.drop_column('models', 'model_version')
