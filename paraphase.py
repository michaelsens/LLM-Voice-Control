#!/usr/bin/env python3
# paraphrase.py

import os
import json
import logging
import re
import time
import uuid
from dotenv import load_dotenv
from openai import OpenAI

# ——— Logging Setup ————————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ——— Load Configuration ————————————————————————————————————————————
load_dotenv()
API_KEY     = os.getenv("OPENAI_API_KEY")
MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o")
INPUT_PATH  = os.getenv("INPUT_PATH", "expanded.jsonl")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "paraphrased.jsonl")
PARAPHRASE_N= int(os.getenv("PARAPHRASE_N", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))

if not API_KEY:
    logger.error("Missing OPENAI_API_KEY in .env")
    exit(1)

client = OpenAI(api_key=API_KEY)

# ——— Prompts —————————————————————————————————————————————————
SYSTEM_PROMPT = (
    "You are a human speaking to a voice-controlled Chrome browser assistant.\n"
    "Given a single base utterance, produce exactly a JSON array of strings:\n"
    "  1) a polite version\n"
    "  2) a curt version\n"
    "  3) a formal version\n"
    "  4) a casual version\n"
    "  5) a version with filler words.\n"
    "Do NOT include any extra text or markdown—just the JSON array."
)
USER_PROMPT = "Base utterance: {utterance}\nGenerate {n} paraphrases in the order specified."

# ——— JSON parsing helper —————————————————————————————————————————————
def parse_json_array(raw: str):
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
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
    raise ValueError(f"Could not parse JSON array from:\n{raw!r}")

# ——— Load expanded dataset —————————————————————————————————————————
with open(INPUT_PATH, "r") as f:
    base_entries = [json.loads(line) for line in f]

seen = set()  # track new utterances
paraphrased = []

# ——— Paraphrase loop —————————————————————————————————————————————
for idx, base in enumerate(base_entries, start=1):
    trace = uuid.uuid4().hex[:8]
    start = time.time()

    prompt = USER_PROMPT.format(utterance=base["utterance"], n=PARAPHRASE_N)
    messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user",   "content": prompt}
    ]

    for attempt in range(MAX_RETRIES+1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            raw = resp.choices[0].message.content or ""
            print(f"\n[Paraphrase {idx} Att {attempt+1}] RAW:\n{raw}\n")

            variants = parse_json_array(raw)
            added = 0
            for variant in variants:
                utt = variant.strip()
                if utt and utt not in seen:
                    seen.add(utt)
                    paraphrased.append({
                        "id":        str(uuid.uuid4()),
                        "parent_id": base["id"],
                        "source":    "paraphrase",
                        "utterance": utt,
                        "rpc":       base["rpc"]
                    })
                    added += 1

            ms = int((time.time()-start)*1000)
            logger.info(f"[{trace}] Added {added}/{len(variants)} variants in {ms}ms")
            break

        except Exception as e:
            logger.warning(f"[{trace}] Attempt {attempt+1} failed: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"[{trace}] Giving up on paraphrasing entry {base['id']}")

# ——— Write out paraphrased dataset —————————————————————————————————————
all_entries = base_entries + paraphrased
with open(OUTPUT_PATH, "w") as f:
    for e in all_entries:
        f.write(json.dumps(e) + "\n")

logger.info(f"Wrote total {len(all_entries)} entries → {OUTPUT_PATH}")
