#!/usr/bin/env python3

import json
import re
import traceback
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from tokenizers import Tokenizer

from models import MiniRPCTransformer

CHECKPOINT = Path("runs/XXX_best_model.pt")
TOKENIZER_PATH = Path("rpc_tokenizer.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = Tokenizer.from_file(str(TOKENIZER_PATH))

PAD_ID = tok.token_to_id("<PAD>")
BOS_ID = tok.token_to_id("<BOS>")
EOS_ID = tok.token_to_id("<EOS>")
VOCAB = tok.get_vocab_size(with_added_tokens=True)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

params = ckpt.get(
    "params",
    {
        "d_model": 256,
        "n_heads": 4,
        "n_enc_layers": 4,
        "n_dec_layers": 4,
        "max_len": 128,
        "dropout": 0.1,
    },
)

model = MiniRPCTransformer(vocab_size=VOCAB, pad_id=PAD_ID, **params).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

print(f"Loaded {CHECKPOINT} • vocab={VOCAB} • device={DEVICE}")

_KV = re.compile(r"(\S+?)=(\S+)")

def dsl_to_rpc(text: str) -> dict:
    body = text.replace("<CMD>", "").replace("</CMD>", "").strip()
    if not body:
        return {}
    method, *rest = body.split()
    params = {k: v.strip('"') for k, v in _KV.findall(" ".join(rest))}
    return {"method": method, "params": params}

@torch.no_grad()
def generate_rpc(utterance: str, max_len: int = 64) -> dict:
    src_ids = torch.tensor(
        [tok.encode(f"<UTT> {utterance} </UTT>").ids], device=DEVICE
    )
    tgt_ids = torch.tensor([[BOS_ID]], device=DEVICE)
    for _ in range(max_len):
        next_id = int(model(src_ids, tgt_ids)[:, -1].argmax(-1))
        tgt_ids = torch.cat(
            [tgt_ids, torch.tensor([[next_id]], device=DEVICE)], dim=1
        )
        if next_id == EOS_ID:
            break
    dsl = tok.decode(tgt_ids.squeeze().tolist())
    return dsl_to_rpc(dsl)

app = Flask(__name__)

@app.route("/infer", methods=["POST"])
def infer():
    utterance = request.json.get("text", "")
    try:
        rpc = generate_rpc(utterance)
        print("INPUT :", utterance)
        print("RPC   :", rpc)
        return jsonify(rpc)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.errorhandler(Exception)
def catch_all(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Server listening on http://127.0.0.1:6006/infer")
    app.run(host="127.0.0.1", port=6006, threaded=True)
