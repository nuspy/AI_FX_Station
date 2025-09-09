#!/usr/bin/env python3
"""
Create sqlite DB file and parent directory based on provided path or configs/default.yaml.

Usage examples:
  python tests/manual_tests/init_db.py --db-path "D:/Projects/ForexGPT/data/forex.db"
  python tests/manual_tests/init_db.py           # tries to read configs/default.yaml and extract sqlite:///... path
  python tests/manual_tests/init_db.py           # if no config path found, falls back to data/forex_diffusion.db

This fixes sqlite3.OperationalError: unable to open database file by ensuring the directory/file exist.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

def find_sqlite_path_in_yaml(yaml_path: Path) -> str | None:
    try:
        import yaml
    except Exception:
        return None
    if not yaml_path.exists():
        return None
    try:
        raw = yaml_path.read_text(encoding="utf-8")
        # quick regex for sqlite:/// or sqlite://// (windows drive may follow)
        m = re.search(r"(sqlite:/{3,4}[^\s'\"\n]+)", raw)
        if m:
            url = m.group(1)
            # strip sqlite:/// prefix
            if url.startswith("sqlite:///"):
                return url[len("sqlite:///"):]
            if url.startswith("sqlite:////"):
                return url[len("sqlite:////"):]
            return url
        # try to load yaml and look for common keys
        y = yaml.safe_load(raw)
        # Depth-first search for sqlite substring
        def dfs(node):
            if isinstance(node, dict):
                for v in node.values():
                    r = dfs(v)
                    if r:
                        return r
            elif isinstance(node, list):
                for it in node:
                    r = dfs(it)
                    if r:
                        return r
            elif isinstance(node, str):
                if "sqlite" in node:
                    if node.startswith("sqlite:///"):
                        return node[len("sqlite:///"):]
                    return node
            return None
        return dfs(y)
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", help="Explicit sqlite file path to create (overrides config)")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config to probe for sqlite URL")
    args = p.parse_args()

    db_path = None
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        cfg = Path(args.config)
        found = find_sqlite_path_in_yaml(cfg)
        if found:
            # handle sqlite:///C:/... style (strip leading / if necessary on Windows)
            fp = found
            # if starts with / and windows drive like /C:/ fix it
            if fp.startswith("/") and re.match(r"^/[A-Za-z]:", fp):
                fp = fp[1:]
            db_path = Path(fp)

    # fallback default if still not determined
    if db_path is None:
        fallback = Path("data") / "forex_diffusion_test.db"
        print(f"No DB path in config; using fallback path: {fallback}")
        db_path = fallback

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # touch file if not exists
        if not db_path.exists():
            db_path.touch()
        # verify writable
        try:
            with db_path.open("a", encoding="utf-8"):
                pass
        except Exception as e:
            print("Created file but cannot write to it. Check filesystem permissions:", e)
            raise SystemExit(1)
        print("Database file ready at:", str(db_path))
    except Exception as e:
        print("Failed to create DB file or parent directory:", e)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
