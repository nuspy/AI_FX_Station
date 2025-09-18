# Repository Guidelines

## Project Structure & Module Organization
ForexMagic's Python package sits under `src/forex_diffusion` with submodules: `data/` (DB adapters), `features/` (indicator pipelines), `training/` (model trainers), `ui/` (PySide6 views), and `utils/` (config and logging helpers). Backtest orchestration lives in `backtest/` and service glue in `services/`. Automation helpers reside in `scripts/`, while YAML/JSON presets are in `configs/`. Tests are under `tests/`, with higher-touch workflows isolated in `tests/manual_tests/`.

## Build, Test, and Development Commands
Bootstrap with Python 3.11+: `python -m venv .venv`, activate, then `pip install -e .`. Train models via `python -m forex_diffusion.training.train_sklearn --symbol EUR/USD --timeframe 15m --horizon 4 --algo ridge --artifacts_dir artifacts`, adjusting parameters suggested in `configs/default.yaml`. Start the GUI with `python scripts/run_gui.py --testserver` to pair with the Tiingo simulator. Background services launch through `python scripts/start_backtest_worker.py --queue backtest`. Run the fast suite using `pytest tests -q`; call manual scenarios individually (`python tests/manual_tests/ml_workflow_check.py`) when external data is available.

## Coding Style & Naming Conventions
Use PEP 8 with 4-space indent and keep lines under 120 characters. Modules ship with `py.typed`, so preserve precise type hints and prefer explicit `TypedDict` or `Protocol` over loose dicts. Adopt `snake_case` for modules and functions, `PascalCase` for classes, and store configuration additions in `configs/` using YAML or JSON so loaders stay consistent.

## Testing Guidelines
Unit tests mirror package names (`tests/test_*.py`); add regression cases whenever DB queries, feature transforms, or UI controllers change. Fixtures belong in `tests/data/` to keep them discoverable. Before pushing, run `pytest --maxfail=1 --disable-warnings`; leave `tests/manual_tests/` for documented, opt-in integration steps.

## Commit & Pull Request Guidelines
Recent commits are short, present-tense statements like `Backtest da risultati molto simili`, often in Italian and under 60 characters. Follow that style and add detail in the body when necessary. Pull requests should link the tracking ticket, summarise behaviour changes, list affected configs or artifacts, and include screenshots or logs for UI and backtest updates. Tag a domain reviewer and note which tests you ran.

## Environment & Secrets
Duplicate `.env.example` to `.env` for local credentials and keep real secrets out of version control. Docker assets under `docker/` and runtime scripts read the same variables, so validate them with `docker compose config` before sharing new defaults. Large datasets should stay in external storage with download guidance captured in `docs/`.
