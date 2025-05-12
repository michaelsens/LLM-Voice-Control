#!/usr/bin/env python
# inference.py  ------------------------------------------------------
import json, re, torch
from tokenizers import Tokenizer                    # ← generic loader
from models import MiniRPCTransformer

# -------- paths -----------------------------------------------------
CKPT_PATH = "runs/XXX_best_model.pt"
TOK_PATH  = "rpc_tokenizer.json"
DEVICE    = "cpu"                                  # "cuda" for GPU

# -------- 1. tokenizer ---------------------------------------------
tok = Tokenizer.from_file(TOK_PATH)                # ← (was SentencePiece…)
PAD_ID = tok.token_to_id("<PAD>")
BOS_ID = tok.token_to_id("<BOS>")
EOS_ID = tok.token_to_id("<EOS>")
VOCAB  = tok.get_vocab_size(with_added_tokens=True)

# -------- 2. checkpoint & params -----------------------------------
ckpt   = torch.load(CKPT_PATH, map_location=DEVICE)
params = ckpt.get("params", {                      # fallback if missing
    "d_model":      256,
    "n_heads":      4,
    "n_enc_layers": 4,
    "n_dec_layers": 4,
    "max_len":      128,
    "dropout":      0.1,
})

# -------- 3. rebuild model -----------------------------------------
model = MiniRPCTransformer(
    vocab_size=VOCAB,
    pad_id=PAD_ID,
    **params,
).to(DEVICE).eval()
model.load_state_dict(ckpt["model"], strict=True)

# -------- 4. DSL ➜ RPC helper --------------------------------------
_kv = re.compile(r"(\S+?)=(\S+)")
def dsl_to_rpc(text: str) -> dict:
    body = text.replace("<CMD>", "").replace("</CMD>", "").strip()
    if not body:
        return {}
    method = body.split()[0]
    params = {k: v.strip('"') for k, v in _kv.findall(body)}
    return {"method": method, "params": params}

# -------- 5. generator ---------------------------------------------
@torch.no_grad()
def generate_rpc(utterance: str, max_out: int = 32) -> dict:
    src = f"<UTT> {utterance} </UTT>"
    src_ids = torch.tensor([tok.encode(src).ids], device=DEVICE)
    tgt_ids = torch.tensor([[BOS_ID]], device=DEVICE)  # start token

    for _ in range(max_out):
        next_id = int(model(src_ids, tgt_ids)[:, -1].argmax(-1))
        tgt_ids = torch.cat([tgt_ids,
                             torch.tensor([[next_id]], device=DEVICE)], dim=1)
        if next_id == EOS_ID:
            break

    return dsl_to_rpc(tok.decode(tgt_ids.squeeze().tolist()))

# -------- 6. CLI ----------------------------------------------------
if __name__ == "__main__":
    print("Type an utterance (empty line to exit)")
    while True:
        try:
            utter = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not utter:
            break
        print("RPC:", json.dumps(generate_rpc(utter), indent=2))
