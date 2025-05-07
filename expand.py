#!/usr/bin/env python3
# expand.py

import os
import json
import time
import logging
import re
import uuid
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ——— Logging Setup —————————————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ——— Load Configuration —————————————————————————————————————————————————
load_dotenv()
API_KEY     = os.getenv("OPENAI_API_KEY")
MODEL       = os.getenv("OPENAI_MODEL",    "gpt-4o")
INPUT_PATH  = Path(os.getenv("INPUT_PATH",  "generated.jsonl"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "expanded.jsonl"))
EXPAND_N    = int(os.getenv("EXPAND_N",     "10"))
NUM_PASSES  = int(os.getenv("NUM_PASSES",   "2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES",  "1"))

if not API_KEY:
    logger.error("Missing OPENAI_API_KEY in .env")
    exit(1)

client = OpenAI(api_key=API_KEY)

# ——— Prompts —————————————————————————————————————————————————————
SYSTEM_PROMPT = """
You are an inventive, human-style voice assistant for Chrome.
Given a single “seed” command (utterance + RPC), invent at least {n} distinct
voice instructions—some simple, some multi-step—and pair each with a valid
JSON-RPC object. Use the seed as inspiration but feel free to propose new
actions, batch sequences, navigations, toggles, waits, etc.

Output *only* a JSON array of objects with two keys:
  - "utterance": string
  - "rpc": object

No extra text or markdown—just the array.
""".strip()

USER_PROMPT = """
Seed command:
  utterance: {base_utterance}
  rpc: {base_rpc_json}

Generate ≥{n} new commands as described.
""".strip()

# ——— JSON parsing helper —————————————————————————————————————————————
def parse_json_array(raw: str):
    # strip code fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1])
    raw = raw.strip()

    # direct parse if it's a JSON array
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return arr
    except json.JSONDecodeError:
        pass

    # fallback: extract first [...] block
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"Could not parse JSON array from:\n{raw!r}")

# ——— Load seed entries —————————————————————————————————————————————
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    seed = [json.loads(line) for line in f]

# track unique utterances to avoid repeats
seen = set(e["utterance"] for e in seed)
expanded = []

# ——— Multi-pass expansion —————————————————————————————————————————
for pass_idx in range(1, NUM_PASSES+1):
    logger.info(f"=== PASS {pass_idx}/{NUM_PASSES} | Pool size {len(seed)} ===")
    new_pool = []
    for idx, entry in enumerate(seed, start=1):
        base_utt = entry["utterance"]
        base_rpc = entry["rpc"]
        trace = uuid.uuid4().hex[:8]
        start_t = time.time()

        prompt = USER_PROMPT.format(
            base_utterance=base_utt.replace('"','\\"'),
            base_rpc_json=json.dumps(base_rpc).replace('"','\\"'),
            n=EXPAND_N
        )
        messages = [
            {"role":"system", "content":SYSTEM_PROMPT.format(n=EXPAND_N)},
            {"role":"user",   "content":prompt}
        ]

        for attempt in range(MAX_RETRIES+1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                raw = resp.choices[0].message.content or ""
                print(f"\n[Pass {pass_idx} | Entry {idx} | Att {attempt+1}] RAW:\n{raw}\n")

                new_cmds = parse_json_array(raw)
                added = 0
                for cmd in new_cmds:
                    utt = cmd.get("utterance","").strip()
                    rpc = cmd.get("rpc")
                    if utt and rpc and utt not in seen:
                        seen.add(utt)
                        expanded.append({"utterance": utt, "rpc": rpc})
                        new_pool.append({"utterance": utt, "rpc": rpc})
                        added += 1

                ms = int((time.time()-start_t)*1000)
                logger.info(f"[{trace}] Added {added}/{len(new_cmds)} in {ms}ms")
                break

            except Exception as e:
                logger.warning(f"[{trace}] Attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES:
                    logger.error(f"[{trace}] Giving up on entry #{idx}")

    seed = new_pool  # next pass works on newly generated commands

# ——— Write out expanded dataset —————————————————————————————————————
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in expanded:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

logger.info(f"Wrote {len(expanded)} entries → {OUTPUT_PATH.resolve()}")
