#!/usr/bin/env python3
"""
llm_server.py  –  logs INPUT and OUTPUT in plain text
POST /infer  { "text": "<utterance>" }
RETURNS      { "method": "...", "params": {...} }
"""

import json, math, torch, traceback
from flask import Flask, request, jsonify
from models import MiniRPCTransformer   # your architecture

# --- IDs and paths -------------------------------------------------------
PAD_ID, BOS_ID, EOS_ID, SHIFT = 0, 1, 2, 3
CHECKPOINT = "best_model.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# --- tokenizer helpers ---------------------------------------------------
def encode(text): return [BOS_ID] + [ord(c) + SHIFT for c in text] + [EOS_ID]
def decode(ids):
    s = []
    for t in ids:
        if t == EOS_ID: break
        if t not in (PAD_ID, BOS_ID): s.append(chr(t - SHIFT))
    return "".join(s)

# --- load model ----------------------------------------------------------
ckpt        = torch.load(CHECKPOINT, map_location=DEVICE)
model       = MiniRPCTransformer(vocab_size=ckpt["vocab"], pad_id=PAD_ID).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print(f"Loaded {CHECKPOINT} on {DEVICE}")

# --- greedy decode -------------------------------------------------------
@torch.no_grad()
def run_model(text, max_len=128):
    src = torch.tensor(encode(text), device=DEVICE)[None, :]
    tgt = torch.tensor([[BOS_ID]],   device=DEVICE)
    for _ in range(max_len):
        nxt = int(model(src, tgt)[0, -1].argmax())
        if nxt == EOS_ID: break
        tgt = torch.cat([tgt, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return decode(tgt[0].tolist())

# --- Flask ----------------------------------------------------------------
app = Flask(__name__)

@app.route("/infer", methods=["POST"])
def infer():
    utterance = request.json.get("text", "")
    raw       = run_model(utterance)

    print("INPUT :", utterance)
    print("OUTPUT:", raw)

    try:
        rpc = json.loads(raw)
        print("RPC   :", rpc)
        return jsonify(rpc)
    except json.JSONDecodeError:
        print("ERROR : model output is not valid JSON")
        return jsonify({"error": "invalid JSON", "raw": raw}), 422

@app.errorhandler(Exception)
def catch_all(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Server listening on http://127.0.0.1:6006/infer")
    app.run(host="127.0.0.1", port=6006)
