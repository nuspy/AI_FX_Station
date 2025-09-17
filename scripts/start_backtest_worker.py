from __future__ import annotations

"""
Start a simple backtest worker that listens for jobs (MVP runs a demo job).
"""

from src.forex_diffusion.backtest.worker import Worker, TrialConfig


def main():
    worker = Worker()
    # Minimal demo config (baseline only)
    cfg = TrialConfig(
        model_name="baseline_rw",
        prediction_type="Baseline",
        timeframe="1m",
        horizons_sec=[60, 300, 900],
        samples_range=(200, 1500, 200),
        indicators={},
        interval={"type": "preset", "preset": "30d", "walkforward": {"train": "90d", "test": "7d", "step": "7d", "gap": "0d"}},
        data_version=None,
    )
    worker.run_job(job_id=0, configs=[cfg])


if __name__ == "__main__":
    main()


