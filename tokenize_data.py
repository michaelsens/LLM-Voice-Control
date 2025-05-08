# tokenise_expanded.py ------------------------------------------
# Usage:
#   python tokenize.py \
#          --jsonl  expanded.jsonl \
#          --train  train.pt \
#          --val    val.pt \
#          --val_ratio 0.10
#
# Requires:  tokenizers >= 0.13  •  torch
# ---------------------------------------------------------------
import argparse, json, random, torch
from pathlib import Path
from tokenizers import SentencePieceBPETokenizer
import json

special = ["<PAD>", "<UTT>", "</UTT>", "<CMD>", "</CMD>",
           "<BOS>", "<EOS>", "<UNK>"]
PAD_ID = 0


def flatten_rpc(rpc) -> str:
    """
    Return a deterministic, 1-line string for **any** RPC variant:
      A) single  {method, params}
      B) batch   {method='batch', params=[{method, params}, …]}
      C) rpc is itself a list
      D) weird items that lack 'method' → dumped as JSON
    """

    # ---------- helper for one command ---------- #
    def one_cmd(cmd):
        if isinstance(cmd, dict) and "method" in cmd:
            m   = cmd["method"]
            p   = cmd.get("params", {})
            if isinstance(p, dict):
                arg_str = " ".join(
                    f"{k}={json.dumps(v, separators=(',',':'))}"
                    for k, v in sorted(p.items())
                )
            else:                       # list / str / int …
                arg_str = json.dumps(p, separators=(',',':'))
            return f"{m} {arg_str}".strip()
        else:                            # fallback
            return json.dumps(cmd, separators=(',',':'))

    # ---------- 4 possible shapes --------------- #
    if isinstance(rpc, list):            # Case C
        return "batch " + " | ".join(one_cmd(c) for c in rpc)

    if isinstance(rpc, dict):
        if rpc.get("method") == "batch" and isinstance(rpc.get("params"), list):
            # Case B
            sub = " | ".join(one_cmd(c) for c in rpc["params"])
            return f"batch {sub}"
        # Case A
        return one_cmd(rpc)

    # fallback for totally unexpected types
    return json.dumps(rpc, separators=(',',':'))

    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",     required=True)
    ap.add_argument("--train",     default="train.pt")
    ap.add_argument("--val",       default="val.pt")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed",      type=int,   default=42)
    args = ap.parse_args()

    # 1) read rows
    rows = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines()]

    # 2) train a small tokenizer
    tok = SentencePieceBPETokenizer()
    tok.train_from_iterator(
        (f"<UTT> {r['utterance']} </UTT>\n"
         f"<BOS> <CMD> {flatten_rpc(r['rpc'])} </CMD> <EOS>" for r in rows),
        vocab_size=8000,
        special_tokens=special)
    PAD_ID = tok.token_to_id("<PAD>")

    # 3) shuffle & split
    random.seed(args.seed)
    random.shuffle(rows)
    val_sz = int(len(rows) * args.val_ratio)
    val_rows, train_rows = rows[:val_sz], rows[val_sz:]

    def to_tensor(row):
        src = f"<UTT> {row['utterance']} </UTT>"
        tgt = f"<BOS> <CMD> {flatten_rpc(row['rpc'])} </CMD> <EOS>"
        return {
            "src": torch.tensor(tok.encode(src).ids, dtype=torch.long),
            "tgt": torch.tensor(tok.encode(tgt).ids, dtype=torch.long),
        }

    train_pt = [to_tensor(r) for r in train_rows]
    val_pt   = [to_tensor(r) for r in val_rows]

    torch.save(train_pt, args.train)
    torch.save(val_pt,   args.val)
    tok.save("rpc_tokenizer.json")

    print(f"✓ tokeniser saved as rpc_tokenizer.json")
    print(f"✓ {len(train_pt)} train  tensors  → {args.train}")
    print(f"✓ {len(val_pt)}   val    tensors  → {args.val}")

if __name__ == "__main__":
    main()
