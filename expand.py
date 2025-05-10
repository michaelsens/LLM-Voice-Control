#!/usr/bin/env python3
# expand.py

import os
import json
import time
import logging
import re
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
INPUT_PATH  = Path(os.getenv("INPUT_PATH",  "gtest.jsonl"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "expanded.jsonl"))
EXPAND_N    = int(os.getenv("EXPAND_N",     "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES",  "1"))

if not API_KEY:
    logger.error("Missing OPENAI_API_KEY in .env")
    exit(1)

client = OpenAI(api_key=API_KEY)

# ——— Prompts —————————————————————————————————————————————————————
SYSTEM_PROMPT = f"""
You are an English-speaking trainer for a Chrome voice assistant.  
You know exactly these RPC methods and their parameter schemas:

• navigate→{{"url":string}}  
• openTab→{{}}  
• closeTab→{{"tabId":number|null}}  
• switchTab→{{"index":number}}  
• click→{{"text":string}}  
• type→{{"text":string,"field":string}}  
• scroll→{{"direction":string,"amount":number}}  
• reload→{{}}  
• search→{{"query":string}}  
• goBack→{{"steps":number}}  
• goForward→{{"steps":number}}

Your goal is to invent **exactly** {EXPAND_N} brand-new, single-step commands **per seed**, using only one of these RPCs each.  
To maximize coverage and human-like variety, be sure to:

- Rotate through all methods across the batch.  
- Use diverse websites (e.g. twitter.com, amazon.com, bbc.com, maps.google.com, youtube.com, gmail.com, docs.google.com).  
- Vary search queries (weather, flights, recipes, stock quotes, translations, tech tutorials, news headlines, “restaurants near me”).  
- Vary “type” fields (search bar, email To, Subject, Body, comment box, chat input, name, address, zip code, promo code, feedback form, password field, username field).  
- Vary scrolls: directions up/down/left/right, numeric amounts (100 px, 300 px, 800 px) and human phrases (“a bit,” “halfway,” “all the way”).  
- Mix tones: curt (“reload”), polite (“could you please reload?”), casual (“hey, refresh this”), questions (“would you mind going back?”).  
- Sprinkle in pleasantries (“please,” “thanks,” “would you mind,” “could you”), synonyms (“hit,” “tap,” “press,” “open,” “visit,” “load,” “head to”).  

**Output only** a JSON array of objects, each with exactly two keys:

[
  {
    "utterance": "…your spoken command…",
    "rpc": {
      "method": "…",
      "params": { "correct param shape" }
    }
  },
  …
]
""".strip()

USER_PROMPT = """
Seed:
  utterance: {base_utt}
  rpc:       {base_rpc}

Now generate exactly {n} new one-step commands, one RPC each, as a JSON array.
""".strip()


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
expanded = []

# ——— One-pass expansion —————————————————————————————————————————————
for idx, entry in enumerate(seeds, start=1):
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
            for cmd in cmds:
                utt = cmd.get("utterance","").strip()
                rpc = cmd.get("rpc")
                key = (utt, json.dumps(rpc, sort_keys=True))
                if utt and rpc and key not in seen:
                    seen.add(key)
                    expanded.append({"utterance": utt, "rpc": rpc})
                    added += 1
            logger.info(f"[{trace}] added {added}/{len(cmds)}")
            break

        except Exception as e:
            logger.warning(f"[{trace}] attempt {attempt+1} failed: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"[{trace}] giving up")

# ——— Write out —————————————————————————————————————————————————————
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in expanded:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

logger.info(f"Wrote {len(expanded)} expanded entries → {OUTPUT_PATH.resolve()}")
