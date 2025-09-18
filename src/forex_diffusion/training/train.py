import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    # Placeholder: integra qui il tuo trainer Lightning reale.
    out_dir = Path(args.artifacts_dir) / "lightning"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{args.symbol.replace('/','')}_{args.timeframe}_h{args.horizon}_ep{args.epochs}.ckpt"
    with open(ckpt, "wb") as f:
        f.write(b"PLACEHOLDER_CHECKPOINT")
    print(f"[OK] lightning placeholder saved: {ckpt}")

if __name__ == "__main__":
    main()
