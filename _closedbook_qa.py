#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct closed-book QA with qwen2.5:7b (no corpus injection).

Records answers in the same format as the LinearRAG combined predictions:
{question, gold_answer, pred_answer, question_set, question_index, global_index}.
Resumable: writes one JSONL line per answered question and skips completed ids.
"""
import json
import os
import sys
import time
import urllib.request

BASE = "/home/lizhaoxuan/KE-project"
OUT_DIR = os.path.join(BASE, "results", "harrypotter", "direct_qwen2.5-7b_closedbook")
OUT_JSONL = os.path.join(OUT_DIR, "closedbook_predictions.jsonl")
OUT_JSON = os.path.join(OUT_DIR, "closedbook_predictions.json")
MODEL = "qwen2.5:7b"
API = "http://127.0.0.1:11434/api/chat"
NUM_CTX = 8192

BOOKS = ["book1", "book2", "book3", "book4", "book5", "book6", "book7"]

SYSTEM = (
    "You are answering factual questions about the seven Harry Potter novels. "
    "Answer concisely, in one short phrase or a single sentence, based on the books. "
    "Do not explain your reasoning."
)

def load_questions():
    """Rebuild the canonical 1000-question order (book1..book7, in-file order)."""
    rows = []
    gi = 0
    for book in BOOKS:
        with open(os.path.join(BASE, "dataset", "question", f"{book}.json"), encoding="utf-8") as f:
            items = json.load(f)
        for qi, item in enumerate(items):
            rows.append({
                "question": item["question"],
                "gold_answer": item.get("answer", ""),
                "question_set": book,
                "question_index": qi,
                "global_index": gi,
            })
            gi += 1
    return rows

def read_done():
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                done.add(rec.get("global_index"))
    return done

def ask(question):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "options": {"num_ctx": NUM_CTX, "temperature": 0.0, "max_tokens": 128},
        "stream": False,
    }
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    content = data.get("message", {}).get("content", "").strip()
    return content

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_questions()
    done = read_done()
    start = time.time()
    ok = 0
    fail = 0
    print(f"total={len(rows)} already_done={len(done)}", flush=True)
    for row in rows:
        gi = row["global_index"]
        if gi in done:
            continue
        try:
            answer = ask(row["question"])
        except Exception as exc:  # network/model hiccup
            fail += 1
            print(f"[{gi}] ERROR {type(exc).__name__}: {exc}", flush=True)
            time.sleep(3)
            continue
        rec = dict(row)
        rec["pred_answer"] = answer
        with open(OUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ok += 1
        if ok % 25 == 0:
            elapsed = time.time() - start
            print(f"progress {ok} done in {elapsed:.0f}s", flush=True)
    # merge into final JSON
    final = []
    seen = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("global_index") not in seen:
                    seen.add(rec.get("global_index"))
                    final.append(rec)
    final.sort(key=lambda r: r["global_index"])
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"finished: answered={len(final)} fail={fail} -> {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
