#!/usr/bin/env python3
# generate.py

import os
import json
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

# ——— Synonym Templates for Common Actions —————————————————————————————————
OPEN_TAB_SYNS = [
    "open a new tab", "open new tab", "create a new tab",
    "start a new tab", "add a tab"
]
CLOSE_TAB_SYNS = [
    "close the current tab", "close this tab", "shut this tab",
    "exit this tab", "close the page"
]
NEW_WINDOW_SYNS = ["open a new window", "open window", "launch a window"]
CLOSE_WINDOW_SYNS = ["close the current window", "close this window"]
RELOAD_SYNS = ["reload the page", "refresh the page", "hit refresh", "reload"]
NAVIGATE_SYNS = [
    "go to {url}", "navigate to {url}", "open {url}",
    "visit {url}", "load {url}"
]
SEARCH_SYNS = [
    "search for {query}", "look up {query}", "find {query}",
    "google {query}", "search {query}"
]
BOOKMARK_SYNS = ["bookmark this page", "add bookmark", "star this page"]
OPEN_BOOKMARKS_SYNS = ["open bookmarks manager", "show bookmarks", "view bookmarks"]
OPEN_HISTORY_SYNS = ["open browsing history", "view history", "show history"]
OPEN_DOWNLOADS_SYNS = ["open downloads page", "show downloads", "view my downloads"]
PIN_TAB_SYNS = ["pin this tab", "add tab to pinned", "keep this tab open"]
MUTE_TAB_SYNS = ["mute this tab", "silence this tab", "turn off tab sound"]
ZOOM_IN_SYNS = ["zoom in", "enlarge page", "increase zoom"]
ZOOM_OUT_SYNS = ["zoom out", "shrink page", "decrease zoom"]
RELOAD_HARD_SYNS = ["hard reload", "force refresh", "reload without cache"]
FIND_SYNS = ["find on page", "search within page", "find text on this page"]

# ——— Generators for Everyday Chrome Actions —————————————————————————————

def gen_new_tab():
    rpc = {"method":"new_tab","params":{}}
    for syn in OPEN_TAB_SYNS:
        yield syn, rpc

def gen_close_tab():
    rpc = {"method":"close_tab","params":{}}
    for syn in CLOSE_TAB_SYNS:
        yield syn, rpc

def gen_new_window():
    rpc = {"method":"new_window","params":{}}
    for syn in NEW_WINDOW_SYNS:
        yield syn, rpc

def gen_close_window():
    rpc = {"method":"close_window","params":{}}
    for syn in CLOSE_WINDOW_SYNS:
        yield syn, rpc

def gen_reload_page():
    rpc = {"method":"reload","params":{}}
    for syn in RELOAD_SYNS:
        yield syn, rpc

def gen_navigate_url():
    sample_sites = ["https://google.com", "https://youtube.com", "https://wikipedia.org"]
    for url in sample_sites:
        rpc = {"method":"navigate","params":{"url":url}}
        for tpl in NAVIGATE_SYNS:
            yield tpl.format(url=url), rpc

def gen_search_omnibox():
    queries = ["weather today", "news headlines", "python tutorials"]
    for q in queries:
        rpc = {"method":"search","params":{"query":q}}
        for tpl in SEARCH_SYNS:
            yield tpl.format(query=q), rpc

def gen_bookmark_page():
    rpc = {"method":"bookmark_page","params":{}}
    for syn in BOOKMARK_SYNS:
        yield syn, rpc

def gen_open_bookmarks():
    rpc = {"method":"open_bookmarks","params":{}}
    for syn in OPEN_BOOKMARKS_SYNS:
        yield syn, rpc

def gen_open_history():
    rpc = {"method":"open_history","params":{}}
    for syn in OPEN_HISTORY_SYNS:
        yield syn, rpc

def gen_open_downloads():
    rpc = {"method":"open_downloads","params":{}}
    for syn in OPEN_DOWNLOADS_SYNS:
        yield syn, rpc

def gen_pin_tab():
    rpc = {"method":"pin_tab","params":{}}
    for syn in PIN_TAB_SYNS:
        yield syn, rpc

def gen_mute_tab():
    rpc = {"method":"mute_tab","params":{}}
    for syn in MUTE_TAB_SYNS:
        yield syn, rpc

def gen_zoom():
    rpc_in  = {"method":"zoom","params":{"direction":"in"}}
    rpc_out = {"method":"zoom","params":{"direction":"out"}}
    for syn in ZOOM_IN_SYNS:
        yield syn, rpc_in
    for syn in ZOOM_OUT_SYNS:
        yield syn, rpc_out

def gen_reload_hard():
    rpc = {"method":"reload_ignore_cache","params":{}}
    for syn in RELOAD_HARD_SYNS:
        yield syn, rpc

def gen_find_on_page():
    rpc = {"method":"find","params":{"query":"<text>"}}
    for syn in FIND_SYNS:
        yield syn, rpc

# ——— Aggregate & Write Dataset —————————————————————————————————————————

generators = [
    gen_new_tab, gen_close_tab, gen_new_window, gen_close_window,
    gen_reload_page, gen_navigate_url, gen_search_omnibox,
    gen_bookmark_page, gen_open_bookmarks, gen_open_history,
    gen_open_downloads, gen_pin_tab, gen_mute_tab,
    gen_zoom, gen_reload_hard, gen_find_on_page
]

seen = set()
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for gen in generators:
        for utt, rpc in gen():
            key = (utt.lower(), json.dumps(rpc, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps({"utterance": utt, "rpc": rpc}, ensure_ascii=False) + "\n")

logger.info(f"Generated {len(seen)} entries → {OUTPUT_PATH.resolve()}")
