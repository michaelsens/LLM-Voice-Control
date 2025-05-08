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
    "gen_navigate":   6,
    "gen_open_tab":   2,
    "gen_close_tab":  2,
    "gen_switch_tab": 2,
    "gen_click":      10,
    "gen_type":       6,
    "gen_scroll":     4,
    "gen_reload":     1,
    "gen_search":     8,
    "gen_go_back":    1,
    "gen_go_forward": 1,
}

# ——— Synonym Templates —————————————————————————————————————————————————————
NAV_SYNS = [
    "go to {url}", "navigate to {url}", "open {url}", "visit {url}", "load {url}",
    "head to {url}", "take me to {url}", "launch {url}", "bring me to {url}",
    "access {url}", "check out {url}", "direct to {url}", "show me {url}", "browse to {url}"
]

OPEN_TAB_SYNS = [
    "open a new tab", "create a new tab", "start a new tab", "launch a new tab",
    "make a new tab", "new browser tab", "add another tab", "begin a new tab"
]

CLOSE_TAB_SYNS = [
    "close the current tab", "close this tab", "exit this tab", "shut this tab",
    "terminate this tab", "remove this tab", "close active tab", "close open tab"
]

# use ordinals for tabs
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]

SWITCH_SYNS = [
    "switch to the {ordinal} tab", "go to the {ordinal} tab", "open the {ordinal} tab",
    "navigate to the {ordinal} tab", "bring up the {ordinal} tab",
    "change to the {ordinal} tab", "jump to the {ordinal} tab"
]

CLICK_SYNS = [
    "click {text}", "press {text}", "tap {text}", "select {text}",
    "hit {text}", "activate {text}", "choose {text}", "pick {text}",
    "tap on {text}", "click on {text}", "push {text}", "engage {text}"
]

TYPE_SYNS = [
    "type {text} into {field}", "enter {text} in {field}", "input {text} into {field}",
    "write {text} in the {field}", "fill in {field} with {text}", "put {text} into {field}",
    "add {text} to {field}", "insert {text} into {field}", "fill {field} with {text}"
]

SCROLL_SYNS = [
    "scroll {direction}", "scroll {direction} by {amount} pixels",
    "move {direction}", "move the page {direction}", "scroll {amount} pixels {direction}",
    "pan {direction}", "shift view {direction}", "slide {direction}"
]

RELOAD_SYNS = [
    "reload the page", "refresh the page", "reload this page",
    "refresh this page", "reload tab", "refresh tab", "refresh the browser"
]

SEARCH_SYNS = [
    "search for {query}", "look up {query}", "find {query}",
    "google {query}", "look for {query}", "search the web for {query}",
    "do a search for {query}", "try searching {query}"
]

BACK_SYNS = [
    "go back", "navigate back", "back", "move back", "back up",
    "return to previous page", "go to last page", "previous page"
]

FORWARD_SYNS = [
    "go forward", "navigate forward", "forward", "move forward",
    "next page", "proceed to next", "advance page"
]


# ——— Generators ———————————————————————————————————————————————————————

def gen_navigate():
    sites = [
        "google.com", "wikipedia.org", "youtube.com", "facebook.com", "reddit.com",
        "twitter.com", "instagram.com", "gmail.com", "amazon.com", "linkedin.com",
        "netflix.com", "stackoverflow.com", "bbc.com", "nytimes.com", "bing.com",
        "weather.com", "espn.com", "twitch.tv", "dropbox.com", "github.com",
        "medium.com", "yahoo.com", "cnn.com", "imdb.com", "apple.com", "zoom.us",
        "spotify.com", "docs.google.com", "drive.google.com", "calendar.google.com",
        "booking.com", "airbnb.com", "paypal.com", "quora.com", "notion.so",
        "canva.com", "web.whatsapp.com", "messenger.com", "maps.google.com",
        "translate.google.com", "udemy.com", "coursera.org", "khanacademy.org"
    ]

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
    targets = [
        "submit", "search", "exit", "login", "logout", "compose", "send", "settings", 
        "profile", "help", "all", "news", "videos", "images", "shopping", "next", 
        "previous", "cancel", "save", "ok", "delete", "upload", "download", "cart", 
        "checkout", "menu", "home", "explore", "watch", "play", "pause", "like", 
        "dislike", "comment", "share", "subscribe", "follow", "unfollow", "edit", 
        "continue", "details", "more", "learn more", "buy now", "get started", 
        "try for free", "see plans", "book now", "contact us", "read more", 
        "join now", "sign up", "view profile", "open chat", "rate", "review"
    ]

    for text in targets:
        for tpl in CLICK_SYNS:
            yield tpl.format(text=text), {"method":"click","params":{"text":text}}

def gen_type():
    fields = [
        "search", "subject", "email body", "username field", "username", "password field", "password",
        "comment box", "chat input", "address form", "feedback form", "contact form",
        "billing info", "credit card number", "city field", "city", "zip code", "zip",
        "promo code", "note field", "note", "support message", "bio", "status update",
        "review text", "to field", "to", "cc field", "cc", "bcc field", "bcc"
    ]


    texts = [
        "hello world", "meeting at 3pm", "openai rocks", "draft email content", 
        "John Doe", "123 Main St", "This is a test", "Lorem ipsum", "myPassword123", 
        "I love this!", "Let’s connect soon", "Great article!", "Please call me back",
        "Need help with this", "Order #12345", "Invoice attached", "Thanks!", 
        "Looking forward to it", "Happy to help", "Best regards", "Test123", 
        "Don't forget the deadline", "Here's the info you asked for", 
        "Let's schedule a meeting", "Join us now!"
    ]

    for field in fields:
        for text in texts:
            for tpl in TYPE_SYNS:
                yield tpl.format(text=text,field=field), {"method":"type","params":{"text":text,"field":field}}

def gen_scroll():
    numeric_scrolls = [
        ("down", 100), ("down", 200), ("down", 300), ("down", 500), ("down", 750), ("down", 1000), ("down", 1500),
        ("up", 100), ("up", 200), ("up", 300), ("up", 500), ("up", 750), ("up", 1000), ("up", 1500),
        ("left", 50), ("left", 100), ("left", 150), ("left", 200), ("left", 300),
        ("right", 50), ("right", 100), ("right", 150), ("right", 200), ("right", 300)
    ]

    for direction, amt in numeric_scrolls:
        for tpl in SCROLL_SYNS:
            # Generate two versions: with and without the word "pixels"
            yield tpl.format(direction=direction, amount=amt), {
                "method": "scroll",
                "params": {"direction": direction, "amount": amt}
            }

            if "pixels" in tpl:
                tpl_no_pixels = tpl.replace(" by {amount} pixels", " by {amount}").replace("{amount} pixels", "{amount}")
                yield tpl_no_pixels.format(direction=direction, amount=amt), {
                    "method": "scroll",
                    "params": {"direction": direction, "amount": amt}
                }

    NATURAL_SCROLL_MAP = {
        "a little": 100,
        "a bit": 150,
        "slightly": 150,
        "halfway": 500,
        "a lot": 800,
        "all the way": 1500,
        "far": 1000
    }

    NATURAL_SCROLL_SYNS = [
        "scroll {direction} a little", "scroll {direction} a bit", "scroll {direction} a lot",
        "scroll {direction} slightly", "scroll {direction} halfway", "scroll {direction} all the way",
        "scroll a bit {direction}", "move slightly {direction}", "move far {direction}"
    ]

    for direction in ["up", "down", "left", "right"]:
        for tpl in NATURAL_SCROLL_SYNS:
            for phrase, mapped_amt in NATURAL_SCROLL_MAP.items():
                if phrase in tpl:
                    yield tpl.format(direction=direction), {
                        "method": "scroll",
                        "params": {"direction": direction, "amount": mapped_amt}
                    }


def gen_reload():
    for tpl in RELOAD_SYNS:
        yield tpl, {"method":"reload","params":{}}

def gen_search():
    queries = [
        "weather today", "news headlines", "python tutorials", "restaurants near me",
        "how to tie a tie", "best laptops 2025", "AI tools for developers", 
        "cheap flights to NYC", "movie showtimes", "top programming languages", 
        "HTML vs CSS", "JavaScript frameworks", "coffee shops nearby", 
        "fitness tips", "remote jobs", "latest tech news", "celebrity gossip", 
        "meditation music", "stock market trends", "NBA scores", "book recommendations",
        "what is machine learning", "translate hello to Spanish", "recipe for pancakes",
        "weather in Tokyo", "is it going to rain", "travel restrictions for Japan"
    ]

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
