#!/usr/bin/env python3
"""Clean the single-column, text-layer PDFs in ../原始数据 using Poppler coordinates."""
import argparse
import collections
import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()

def extract(path):
    proc = subprocess.run(["pdftotext", "-bbox-layout", "-enc", "UTF-8",
                           str(path), "-"], capture_output=True, check=True)
    root = ET.fromstring(proc.stdout)
    pages = []
    for page in root.findall(".//{*}page"):
        lines = []
        for el in page.findall(".//{*}line"):
            text = norm(" ".join(w.text or "" for w in el))
            if text:
                lines.append(dict(text=text, **{k: float(el.get(k)) for k in
                             ("xMin", "yMin", "xMax", "yMax")}))
        pages.append((float(page.get("width")), float(page.get("height")),
                      sorted(lines, key=lambda l: (l["yMin"], l["xMin"]))))
    return pages

def upper(text):
    letters = "".join(c for c in text if c.isalpha())
    return bool(letters) and letters.isupper() and 3 <= len(text) <= 100 and not text.startswith(('“', '"', '‘'))

def signature(text):
    return re.sub(r"\d+", "#", text.lower())

def clean(path, args):
    pages = extract(path)
    sizes = collections.Counter(round(l["yMax"]-l["yMin"], 1)
                                for _, _, ls in pages for l in ls if len(l["text"]) > 30)
    if not sizes:
        raise ValueError("No text layer detected; OCR is required.")
    body_height = sizes.most_common(1)[0][0]
    repeated = collections.Counter()
    for w, h, ls in pages:
        repeated.update(set(signature(l["text"]) for l in ls
                            if l["yMin"] < h*.06 or l["yMin"] > h*.95))
    repeated = {s for s, n in repeated.items() if n >= max(3, len(pages)*.15)}
    removed, chapters, paragraphs = [], [], []
    buffer = ""
    previous = None

    def flush():
        nonlocal buffer
        if buffer:
            paragraphs.append(buffer.strip())
            buffer = ""

    def discard(page, line, reason):
        removed.append({"page":page, "reason":reason, "text":line["text"]})

    for pn, (width, height, raw) in enumerate(pages, 1):
        lines = []
        footnote = False
        for line in raw:
            t, y = line["text"], line["yMin"]
            small = line["yMax"]-y < body_height*.9
            marginal = y < height*.06 or y > height*.95
            if marginal and (signature(t) in repeated or
                             re.fullmatch(r"(?:Page\s*\|?\s*)?\d+", t, re.I)):
                discard(pn, line, "running_header_or_footer")
                continue
            if y > height*.95 and (re.match(r"Page\s*\|", t, re.I) or "Rowling" in t):
                discard(pn, line, "page_footer")
                continue
            if y > height*.72 and small and re.match(r"^(?:\d{1,3}[.)]?|[*†‡])\s+", t):
                footnote = True
            if footnote and small:
                discard(pn, line, "footnote")
                continue
            if args.drop_regex and any(re.search(pat, t) for pat in args.drop_regex):
                discard(pn, line, "custom_regex")
                continue
            lines.append(line)
        if not lines:
            continue
        first = lines[0]
        # This edition places chapter headings well below the top, above the body.
        is_heading = (upper(first["text"]) and height*.20 < first["yMin"] < height*.65
                      and abs((first["xMin"]+first["xMax"])/2-width/2) < width*.15)
        if is_heading:
            flush()
            title_lines = [lines.pop(0)]
            while lines and upper(lines[0]["text"]) and lines[0]["yMin"]-title_lines[-1]["yMax"] < body_height*1.5:
                title_lines.append(lines.pop(0))
            title = " ".join(l["text"] for l in title_lines)
            paragraphs.append(title)
            chapters.append({"title":title, "start_page":pn})
            previous = None
        if not chapters:
            for line in lines:
                discard(pn, line, "front_matter")
            continue
        for line in lines:
            if previous:
                same_page = previous[0] == pn
                gap = line["yMin"]-previous[1]["yMax"]
                if same_page and gap > body_height*.65:
                    flush()
                elif not same_page and buffer and re.search(r'[.!?…][”\'"]?$', buffer):
                    flush()
            t = line["text"]
            # Preserve lexical hyphens (e.g. low-growing); never guess spelling.
            if buffer.endswith("-"):
                buffer += t
            else:
                buffer = (buffer + " " + t).strip()
            previous = (pn, line)
    flush()
    if not chapters:
        raise ValueError("No chapter titles detected: this PDF does not match the Book layout.")
    for i, ch in enumerate(chapters):
        ch["end_page"] = chapters[i+1]["start_page"]-1 if i+1 < len(chapters) else len(pages)
    text = "\n\n".join(paragraphs)+"\n"
    return text, {"source":str(path.resolve()), "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                  "pages":len(pages), "chapters":chapters, "characters":len(text),
                  "removed_counts":dict(collections.Counter(r["reason"] for r in removed)),
                  "removed_lines":removed,
                  "warnings":["Heuristic layout cleanup; review chapter titles and removed_lines.",
                              "Raster-only pages are not OCRed; embedded images are not extracted.",
                              "Line-end hyphens are preserved to avoid changing compound words."]}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT/"原始数据", help="PDF file or directory")
    p.add_argument("--output", type=Path, default=ROOT/"cleanUP"/"txt")
    p.add_argument("--drop-regex", action="append", default=[], help="Additional unwanted line regex (repeatable)")
    p.add_argument("--overwrite", action="store_true", help="Replace existing generated files")
    args = p.parse_args()
    for pattern in args.drop_regex:
        re.compile(pattern)
    if not shutil.which("pdftotext"):
        p.error("Missing pdftotext. Install with: sudo apt-get install poppler-utils")
    files = ([args.input] if args.input.is_file() and args.input.suffix.lower()==".pdf" else
             sorted(f for f in args.input.glob("*") if f.suffix.lower()==".pdf" and f.is_file()))
    if not files:
        p.error("No PDF files found.")
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for source in files:
        dest = args.output/(source.stem+".txt")
        report = args.output/(source.stem+".report.json")
        if not args.overwrite and (dest.exists() or report.exists()):
            print(f"SKIP existing output: {dest}", file=sys.stderr)
            failed = True
            continue
        try:
            text, metadata = clean(source, args)
            dest.write_text(text, encoding="utf-8")
            report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            print(f"OK {source.name}: {metadata['pages']} pages, {len(metadata['chapters'])} chapters -> {dest}", flush=True)
        except (ValueError, OSError, subprocess.CalledProcessError, ET.ParseError) as exc:
            print(f"ERROR {source}: {exc}", file=sys.stderr)
            failed = True
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
