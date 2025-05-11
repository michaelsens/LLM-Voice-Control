#!/usr/bin/env python3
# expand.py

import os
import json
import time
import logging
import re
import argparse
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

# ——— CLI Arguments —————————————————————————————————————————————————————
parser = argparse.ArgumentParser(description="Expand JSONL seeds via OpenAI and resumeable output")
parser.add_argument(
    "--start-line",
    type=int,
    default=None,
    help="Seed index to start at (1-based). If omitted, reads from .expand_resume or starts at 1."
)
args = parser.parse_args()

# ——— Load Configuration —————————————————————————————————————————————————
load_dotenv()
API_KEY     = os.getenv("OPENAI_API_KEY")
MODEL       = os.getenv("OPENAI_MODEL",    "gpt-4o")
INPUT_PATH  = Path(os.getenv("INPUT_PATH",  "expanded.jsonl"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "expanded-2.jsonl"))
RESUME_PATH = Path(os.getenv("RESUME_PATH", ".expand_resume"))
EXPAND_N    = int(os.getenv("EXPAND_N",     "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES",  "1"))

if not API_KEY:
    logger.error("Missing OPENAI_API_KEY in .env")
    exit(1)

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = f"""
You are an English-speaking trainer for a Chrome voice assistant.  
You are allowed to generate exactly these RPC methods and their parameter schemas no matter the seed:

• navigate→{{"url":string}}  
• click→{{"text":string}}  
• type→{{"text":string,"field":string}}  
• search→{{"query":string}}  

Your goal is to invent **exactly** {EXPAND_N} brand-new, single-step commands **per seed**, using only one of these RPCs each.  
To maximize coverage and human-like variety, be sure to:

- Rotate through all methods across the batch.  
- Use diverse websites (e.g. twitter.com, amazon.com, bbc.com, maps.google.com, youtube.com, gmail.com, docs.google.com and more, emphasis on more).  
- Vary search queries (weather, flights, recipes, stock quotes, translations, tech tutorials, news headlines, “restaurants near me” and more, emphasis on more, what someone would search on avereage navigating the internet).  
- Vary “type” fields (search bar, email To, Subject, Body, comment box, chat input, name, address, zip code, promo code, feedback form, password, username and more).  
  **Always drop the word “field”**—e.g. use `"field":"username"`, not `"username field"` do the same with similar descriptors.  
  **For type commands**, utterances **must** include the exact literal text to type (e.g. `type example@example.com into email`)—**never** use vague placeholders like “my email” or “your address.”  
- Mix tones (casual, formal, professional, friendly, etc.) and styles (concise, verbose, direct, indirect, etc.).
- Sprinkle in pleasantries (“please,” “thanks,” “would you mind,” “could you”, and more), synonyms (“hit,” “tap,” “press,” “open,” “visit,” “load,” “head to”, and more).
- **Do not** wrap any visible text in quotation marks in the utterance—utter raw words or phrases.  
  Use apostrophes only for normal contractions (e.g. “don’t”).  

**Output only** a JSON array of objects, each with exactly two keys:

[
  {{
    "utterance": "…your spoken command…",
    "rpc": {{
      "method": "…",
      "params": {{ "…" }}
    }}
  }},
  …
]
""".strip()

USER_PROMPT = """
Seed:
  utterance: {base_utt}
  rpc:       {base_rpc}

Now generate exactly {n} new one-step commands, one RPC each, as a JSON array.
""".strip()

# ——— Determine where to start —————————————————————————————————————————————
if args.start_line is not None:
    start_idx = args.start_line
    logger.info(f"Starting at user-specified seed #{start_idx}")
else:
    if RESUME_PATH.exists():
        last = int(RESUME_PATH.read_text().strip() or "0")
        start_idx = last + 1
        logger.info(f"Resuming from previous run: starting at seed #{start_idx}")
    else:
        start_idx = 1
        logger.info("No resume file found; starting at seed #1")

# ——— JSON parsing helper —————————————————————————————————————————————
def parse_json_array(raw: str):
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    raw = raw.strip()
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return arr
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Could not parse JSON array:\n{raw!r}")

# ——— Load seeds —————————————————————————————————————————————————————
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    seeds = [json.loads(line) for line in f]

seen = set()

# ——— Open output file in append mode —————————————————————————————————————
with open(OUTPUT_PATH, "a", encoding="utf-8") as fout:
    # ——— Process each seed ——————————————————————————————————————
    for idx, entry in enumerate(seeds, start=1):
        if idx < start_idx:
            continue

        base_utt = entry["utterance"]
        base_rpc = entry["rpc"]
        trace    = f"#{idx:03d}"
        usr_msg  = USER_PROMPT.format(
            base_utt=base_utt.replace('"','\\"'),
            base_rpc=json.dumps(base_rpc).replace('"','\\"'),
            n=EXPAND_N
        )
        messages = [
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user",   "content":usr_msg}
        ]

        # ——— Try the request (with retries) ————————————————————————
        for attempt in range(MAX_RETRIES+1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.8,
                    top_p=0.9,
                    presence_penalty=0.6
                )
                raw = resp.choices[0].message.content or ""
                logger.info(f"[{trace}] raw response:\n{raw}")
                cmds = parse_json_array(raw)

                added = 0
                # ——— Write each unique command immediately ——————————
                for cmd in cmds:
                    utt = cmd.get("utterance","").strip()
                    rpc = cmd.get("rpc")
                    key = (utt, json.dumps(rpc, sort_keys=True))
                    if utt and rpc and key not in seen:
                        seen.add(key)
                        line = json.dumps({"utterance": utt, "rpc": rpc}, ensure_ascii=False)
                        fout.write(line + "\n")
                        fout.flush()
                        added += 1

                logger.info(f"[{trace}] added {added}/{len(cmds)} commands")
                break

            except Exception as e:
                logger.warning(f"[{trace}] attempt {attempt+1} failed: {e}")
                if attempt == MAX_RETRIES:
                    logger.error(f"[{trace}] giving up on this seed")

        # ——— Update resume file —————————————————————————————————
        RESUME_PATH.write_text(str(idx))
        logger.info(f"Finished seed #{idx}; resume checkpoint updated")

logger.info(f"All done (processed up to seed #{idx}) → {OUTPUT_PATH.resolve()}")