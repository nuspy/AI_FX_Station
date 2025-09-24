# Repository Guidelines

## Project Structure & Module Organization
- Core package under `src/forex_diffusion/`; treat it as the default import root.
- `data/` covers DB adapters; `features/` assembles indicator pipelines.
- `training/` houses trainers; `ui/` presents PySide6 views; `utils/` maintains config/log helpers.
- Backtests run from `backtest/`; background services stay in `services/`.
- Automation lives in `scripts/`; presets and overrides belong in `configs/`.
- Tests mirror the package in `tests/`, with opt-in workflows in `tests/manual_tests/`.
- Persist experiment artifacts inside `artifacts/` or ticket-specific folders for reproducibility.

## Build, Test, and Development Commands
- `python -m venv .venv` then `pip install -e .` bootstraps the editable environment.
- `python -m forex_diffusion.training.train_sklearn --symbol EUR/USD --timeframe 15m --horizon 4 --algo ridge --artifacts_dir artifacts` trains a reference ridge model; tune knobs per `configs/default.yaml`.
- `python scripts/run_gui.py --testserver` launches the PySide6 GUI against the Tiingo simulator.
- `python scripts/start_backtest_worker.py --queue backtest` activates the async backtest worker.
- `pytest tests -q` covers the fast suite; before pushing run `pytest --maxfail=1 --disable-warnings`.

## Coding Style & Naming Conventions
- Follow PEP 8, four-space indent, <120 character lines, and keep explicit typing (package ships with `py.typed`).
- Use `snake_case` for modules/functions and `PascalCase` for classes; maintain TypedDict/Protocol contracts.
- Prefer self-documenting names; add targeted comments only where logic is non-obvious.
- Keep YAML/JSON configs in `configs/` so shared loaders remain consistent.

## Testing Guidelines
- Place unit tests in `tests/test_*.py`; mirror the module structure when adding cases.
- Share fixtures via `tests/data/`; document manual scenarios beside `tests/manual_tests/`.
- Capture regressions for DB adapters, feature transforms, and UI controllers whenever behaviour shifts.
- Use markers sparingly and explain slow/manual cases inline.

## Commit & Pull Request Guidelines
- Keep commit subjects short, present tense, under 60 chars (recent history uses Italian, e.g., `Backtest da risultati molto simili`).
- PRs should link the tracking ticket, summarise behaviour changes, list impacted configs/artifacts, and attach GUI/backtest evidence.
- Note which test commands ran and tag the relevant domain reviewer for faster triage.

## Security & Configuration Tips
- Copy `.env.example` to `.env`; never commit secrets or real API keys.
- Validate Docker updates with `docker compose config` before sharing defaults.
- Store large datasets externally and document retrieval steps in `docs/`.
