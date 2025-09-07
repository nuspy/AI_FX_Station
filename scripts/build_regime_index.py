#!/usr/bin/env python3
"""
Build regime clusters and ANN index from latents in DB.

Usage:
  python scripts/build_regime_index.py --db-url sqlite:///./tmp.db --n_clusters 8 --limit 20000
"""
import argparse
from sqlalchemy import create_engine
from src.forex_diffusion.services.regime_service import RegimeService

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db-url", required=False, default=None)
    p.add_argument("--n_clusters", type=int, default=8)
    p.add_argument("--limit", type=int, default=20000)
    return p.parse_args()

def main():
    args = parse_args()
    engine = None
    if args.db_url:
        engine = create_engine(args.db_url, future=True)
    rs = RegimeService(engine=engine)
    rs.fit_clusters_and_index(n_clusters=args.n_clusters, limit=args.limit)
    print("Regime index built and saved.")

if __name__ == "__main__":
    main()
