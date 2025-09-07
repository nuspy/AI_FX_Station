# alembic env.py - minimal integration with project config
from __future__ import annotations

import sys
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# add project src to path
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.realpath(os.path.join(here, ".."))
sys.path.append(os.path.join(project_root, ".."))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import project config to override SQLAlchemy URL
try:
    from src.forex_diffusion.utils.config import get_config
    cfg = get_config()
    db_url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
    if db_url:
        config.set_main_option("sqlalchemy.url", db_url)
except Exception:
    # fallback: use whatever is in alembic.ini
    db_url = config.get_main_option("sqlalchemy.url")
# alembic env.py - minimal integration with project config
from __future__ import annotations

import sys
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# add project src to path
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.realpath(os.path.join(here, ".."))
sys.path.append(os.path.join(project_root, ".."))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import project config to override SQLAlchemy URL
try:
    from src.forex_diffusion.utils.config import get_config
    cfg = get_config()
    db_url = getattr(cfg.db, "database_url", None) or (cfg.db.get("database_url") if isinstance(cfg.db, dict) else None)
    if db_url:
        config.set_main_option("sqlalchemy.url", db_url)
except Exception:
    # fallback: use whatever is in alembic.ini
    db_url = config.get_main_option("sqlalchemy.url")

target_metadata = None  # we use explicit op.create_table in migrations

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
target_metadata = None  # we use explicit op.create_table in migrations

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
