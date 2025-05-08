# split_data.py
# ------------------------------------------------------------
# Usage:
#   python split_data.py \
#          --in_file  expanded.jsonl \
#          --train_out train.pt \
#          --val_out   val.pt \
#          --val_ratio 0.10
import argparse, json, random, torch
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file",   required=True)          # expanded.jsonl
    ap.add_argument("--train_out", default="train.pt")
    ap.add_argument("--val_out",   default="val.pt")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed",      type=int,   default=42)
    args = ap.parse_args()

    # 1) stream-read every line
    data = []
    with open(args.in_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {i}: {e}") from None

    # 2) shuffle & split
    random.seed(args.seed)
    random.shuffle(data)
    val_size = int(len(data) * args.val_ratio)
    val_data   = data[:val_size]
    train_data = data[val_size:]

    # 3) save as .pt (pickle via torch.save)
    torch.save(train_data, args.train_out)
    torch.save(val_data,   args.val_out)

    print(f"✓ {len(train_data)} train   → {args.train_out}")
    print(f"✓ {len(val_data)}   val     → {args.val_out}")

if __name__ == "__main__":
    main()