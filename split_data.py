# split_expanded.py
# ------------------------------------------------------------
# Usage:
#   python split_data.py \
#          --in_file  expanded.jsonl \
#          --train_out train.pt \
#          --val_out   val.pt \
#          --val_ratio 0.10
import argparse, json, random, torch
from pathlib import Path
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file",  required=True, help="expanded.jsonl")
    ap.add_argument("--train_out", default="train.pt")
    ap.add_argument("--val_out",   default="val.pt")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed",      type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # 1) load the whole .jsonl as a HF Dataset (fast + memory-efficient)
    ds = load_dataset("json", data_files={"all": args.in_file})["all"]

    # 2) shuffle deterministically, then split
    ds = ds.shuffle(seed=args.seed)
    val_size = int(len(ds) * args.val_ratio)

    train_ds = ds.select(range(len(ds) - val_size))
    val_ds   = ds.select(range(len(ds) - val_size, len(ds)))

    # 3) save to .pt (pickled Python list of dicts)
    torch.save(train_ds.to_list(), args.train_out)
    torch.save(val_ds.to_list(),   args.val_out)

    print(f"✓ Saved {len(train_ds)} train samples  → {args.train_out}")
    print(f"✓ Saved {len(val_ds)}   val  samples  → {args.val_out}")

if __name__ == "__main__":
    main()
