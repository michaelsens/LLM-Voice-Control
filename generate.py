#!/usr/bin/env python3
# generate.py

import os
import json
import random
import logging
from pathlib import Path
from dotenv import load_dotenv

# ——— Logging Setup —————————————————————————————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ——— Load Configuration —————————————————————————————————————————————————
load_dotenv()
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "generated.jsonl"))
SAMPLES_PER_WEIGHT = int(os.getenv("SAMPLES_PER_WEIGHT", "10"))

# ——— Weight multipliers for current RPCs —————————————————————————————————————
WEIGHTS = {
    "gen_navigate":   3,
    "gen_open_tab":   2,
    "gen_close_tab":  2,
    "gen_switch_tab": 2,
    "gen_click":      5,
    "gen_type":       3,
    "gen_scroll":     2,
    "gen_reload":     1,
    "gen_search":     4,
    "gen_go_back":    1,
    "gen_go_forward": 1,
}

# ——— Synonym Templates —————————————————————————————————————————————————————
NAV_SYNS      = ["go to {url}", "navigate to {url}", "open {url}", "visit {url}", "load {url}"]
OPEN_TAB_SYNS = ["open a new tab", "create a new tab", "start a new tab"]
CLOSE_TAB_SYNS= ["close the current tab", "close this tab", "exit this tab"]
# use ordinals for tabs
ORDINALS      = ["first", "second", "third", "fourth", "fifth"]
SWITCH_SYNS   = ["switch to the {ordinal} tab", "go to the {ordinal} tab", "open the {ordinal} tab"]
CLICK_SYNS    = ["click {text}", "press {text}", "tap {text}", "select {text}"]
TYPE_SYNS     = ["type {text} into {field}", "enter {text} in {field}", "input {text} into {field}"]
SCROLL_SYNS   = ["scroll {direction}", "scroll {direction} by {amount} pixels"]
RELOAD_SYNS   = ["reload the page", "refresh the page"]
SEARCH_SYNS   = ["search for {query}", "look up {query}", "find {query}"]
BACK_SYNS     = ["go back", "navigate back", "back"]
FORWARD_SYNS  = ["go forward", "navigate forward", "forward"]

# ——— Generators ———————————————————————————————————————————————————————

def gen_navigate():
    sites = ["google.com", "wikipedia.org", "example.com", "youtube.com", "facebook.com", "reddit.com", "twitter.com", "instagram.com"]
    for url in sites:
        for tpl in NAV_SYNS:
            yield tpl.format(url=url), {"method":"navigate","params":{"url":url}}

def gen_open_tab():
    for tpl in OPEN_TAB_SYNS:
        yield tpl, {"method":"openTab","params":{}}

def gen_close_tab():
    for tpl in CLOSE_TAB_SYNS:
        yield tpl, {"method":"closeTab","params":{"tabId":None}}

def gen_switch_tab():
    # ordinal wording, index stays 0-based
    for idx, ord in enumerate(ORDINALS):
        for tpl in SWITCH_SYNS:
            yield tpl.format(ordinal=ord), {"method":"switchTab","params":{"index":idx}}

def gen_click():
    targets = ["submit","search","exit","login","compose","send","settings","profile","help","all","news","videos","images","shopping"]
    for text in targets:
        for tpl in CLICK_SYNS:
            yield tpl.format(text=text), {"method":"click","params":{"text":text}}

def gen_type():
    fields = ["search bar","subject","email body"]
    texts  = ["hello world","meeting at 3pm","openai rocks","draft email content"]
    for field in fields:
        for text in texts:
            for tpl in TYPE_SYNS:
                yield tpl.format(text=text,field=field), {"method":"type","params":{"text":text,"field":field}}

def gen_scroll():
    for direction,amt in [("down",300),("up",300),("left",200),("right",200)]:
        for tpl in SCROLL_SYNS:
            yield tpl.format(direction=direction,amount=amt), {"method":"scroll","params":{"direction":direction,"amount":amt}}

def gen_reload():
    for tpl in RELOAD_SYNS:
        yield tpl, {"method":"reload","params":{}}

def gen_search():
    queries = ["weather today","news headlines","python tutorials"]
    for q in queries:
        for tpl in SEARCH_SYNS:
            yield tpl.format(query=q), {"method":"search","params":{"query":q}}

def gen_go_back():
    for tpl in BACK_SYNS:
        yield tpl, {"method":"goBack","params":{"steps":1}}

def gen_go_forward():
    for tpl in FORWARD_SYNS:
        yield tpl, {"method":"goForward","params":{"steps":1}}

# ——— Aggregate, sample proportionally, and write —————————————————————————————

generators = {
    "gen_navigate":   gen_navigate,
    "gen_open_tab":   gen_open_tab,
    "gen_close_tab":  gen_close_tab,
    "gen_switch_tab": gen_switch_tab,
    "gen_click":      gen_click,
    "gen_type":       gen_type,
    "gen_scroll":     gen_scroll,
    "gen_reload":     gen_reload,
    "gen_search":     gen_search,
    "gen_go_back":    gen_go_back,
    "gen_go_forward": gen_go_forward,
}

random.seed(42)
total = 0
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for name, gen in generators.items():
        weight = WEIGHTS.get(name, 1)
        examples = list(gen())
        target_count = weight * SAMPLES_PER_WEIGHT
        chosen = examples if len(examples) <= target_count else random.sample(examples, target_count)
        for utt, rpc in chosen:
            for k,v in rpc["params"].items():
                if v is None:
                    rpc["params"][k] = None
            f.write(json.dumps({"utterance": utt, "rpc": rpc}, ensure_ascii=False) + "\n")
            total += 1

logger.info(f"Generated {total} examples → {OUTPUT_PATH.resolve()}")
