#!/usr/bin/env python3
from __future__ import annotations

import json, random, re
from pathlib import Path
from typing import Dict, List

INPUT_PATH = Path("expanded-2.jsonl")
OUTPUT_PATH = INPUT_PATH.with_name("adjusted_data.jsonl")

NUMBER_WORDS = {0:"zero",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",16:"sixteen",17:"seventeen",18:"eighteen",19:"nineteen",20:"twenty",21:"twenty one",22:"twenty two",23:"twenty three",24:"twenty four",25:"twenty five"}
WORD_TO_NUM = {v:k for k,v in NUMBER_WORDS.items()}

RE_NUMERIC = re.compile(r"\b([1-9][0-9]*)\b")
RE_NUMBER_WORDS = re.compile(r"\b("+"|".join(re.escape(w) for w in sorted(WORD_TO_NUM,key=len,reverse=True))+r")\b",re.I)
RE_SURROUNDING_QUOTES = re.compile(r"(?<!\w)'([^']+?)'(?!\w)")
TAB_NUMBER_RE = re.compile(r"\btab\s+number\b",re.I)

STATIC_METHOD_COUNTS = {'openTab':10,'closeTab':10,'reload':7, 'switchTab':3}

def load_records(p:Path)->List[Dict]:
    with p.open(encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def save_records(p:Path,recs:List[Dict])->None:
    with p.open('w',encoding='utf-8') as f:
        for r in recs:
            json.dump(r,f,ensure_ascii=False)
            f.write('\n')

def replace_number(u:str,n:int)->str:
    m=RE_NUMERIC.search(u)
    if m:
        return u[:m.start()]+str(n)+u[m.end():]
    m=RE_NUMBER_WORDS.search(u)
    if m:
        w=m.group(1)
        rep=NUMBER_WORDS[n].title() if w[0].isupper() else NUMBER_WORDS[n]
        return u[:m.start()]+rep+u[m.end():]
    return u

def clean(u:str)->str:
    u=RE_SURROUNDING_QUOTES.sub(lambda m:m.group(1),u)
    return u[:1].lower()+u[1:] if u else u

def deep(obj):
    return json.loads(json.dumps(obj))

def expand_numeric(recs:List[Dict])->List[Dict]:
    out=[]
    extra=[5,10,15,20,25,30,50,75,150,250,750]
    scroll=list(range(100,2001,100))+extra
    for r in recs:
        m=r['rpc']['method']
        u=r['utterance']
        has=bool(RE_NUMERIC.search(u) or RE_NUMBER_WORDS.search(u))
        if m=='switchTab' and has:
            for i in range(1,26):
                n=deep(r)
                n['utterance']=replace_number(u,i)
                n['rpc']['params']['index']=i-1
                out.append(n)
                if TAB_NUMBER_RE.search(n['utterance']):
                    alt=deep(n)
                    alt['utterance']=TAB_NUMBER_RE.sub('tab',alt['utterance'])
                    alt['utterance']=re.sub(r'\s{2,}',' ',alt['utterance']).strip()
                    out.append(alt)
            continue
        if m in {'goBack','goForward'} and has:
            for i in range(1,16):
                n=deep(r)
                n['utterance']=replace_number(u,i)
                n['rpc']['params']['steps']=i
                out.append(n)
            continue
        if m=='scroll' and has:
            for amount in scroll:
                n=deep(r)
                n['utterance']=replace_number(u,amount)
                n['rpc']['params']['amount']=amount
                out.append(n)
            continue
        out.append(r)
    return out

def duplicate_static(recs:List[Dict])->List[Dict]:
    o=[]
    for r in recs:
        t=STATIC_METHOD_COUNTS.get(r['rpc']['method'],1)
        for _ in range(t):
            o.append(deep(r))
    return o

def dedup(recs:List[Dict])->List[Dict]:
    s=set()
    u=[]
    for r in recs:
        k=json.dumps(r,sort_keys=True)
        if k not in s:
            s.add(k)
            u.append(r)
    return u

def transform(recs:List[Dict])->List[Dict]:
    recs=expand_numeric(recs)
    for r in recs:
        r['utterance']=clean(r['utterance'])
    recs=dedup(recs)
    recs=duplicate_static(recs)
    random.shuffle(recs)
    return recs

def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file '{INPUT_PATH}' not found.")
    save_records(OUTPUT_PATH,transform(load_records(INPUT_PATH)))

if __name__=='__main__':
    main()
