# Use official Python 3.12 slim image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install basic build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (pip install -e . will read pyproject.toml)
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY configs configs
COPY src src
COPY scripts scripts
COPY migrations migrations
COPY docker docker

# Upgrade pip and install project in editable mode
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e .

# Ensure entrypoint is executable
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8000
# Use official Python 3.12 slim image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install basic build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (pip install -e . will read pyproject.toml)
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY configs configs
COPY src src
COPY scripts scripts
COPY migrations migrations
COPY docker docker

# Upgrade pip and install project in editable mode
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e .

# Ensure entrypoint is executable
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8000

# Entrypoint runs alembic upgrade head and starts uvicorn
ENTRYPOINT ["/app/docker/entrypoint.sh"]
# Entrypoint runs alembic upgrade head and starts uvicorn
ENTRYPOINT ["/app/docker/entrypoint.sh"]
