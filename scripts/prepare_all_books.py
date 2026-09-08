#!/usr/bin/env python3
"""Build one LinearRAG JSON dataset from all cleaned Harry Potter books."""

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "cleanUP" / "txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "harrypotter"
DEFAULT_MODEL = PROJECT_ROOT / "model" / "all-mpnet-base-v2"
TITLE_CORRECTIONS = {
    "THE VANASHIG GLASS": "THE VANISHING GLASS",
    "NICHOLAS FLAMEL": "NICOLAS FLAMEL",
}

@dataclass
class Chapter:
    book_number: int
    chapter_number: int
    title: str
    text: str


@dataclass
class ChunkInfo:
    book_number: int
    chapter_number: int
    chapter_title: str
    chunk_in_chapter: int
    token_count: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_book(report_path: Path) -> Tuple[List[Chapter], Dict[str, object]]:
    match = re.match(r"Book\s+(\d+)\s+-", report_path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot determine book number from {report_path.name}")
    book_number = int(match.group(1))
    txt_path = report_path.with_name(report_path.name.replace(".report.json", ".txt"))
    if not txt_path.is_file():
        raise FileNotFoundError(txt_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    titles = [chapter["title"] for chapter in report["chapters"]]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", txt_path.read_text(encoding="utf-8")) if part.strip()]

    chapters: List[Chapter] = []
    current_title = None
    current_body: List[str] = []
    expected = 0
    for paragraph in paragraphs:
        if expected < len(titles) and paragraph == titles[expected]:
            if current_title is not None:
                chapters.append(Chapter(book_number, len(chapters) + 1, TITLE_CORRECTIONS.get(current_title, current_title), "\n\n".join(current_body)))
            current_title = paragraph
            current_body = []
            expected += 1
        elif current_title is not None:
            current_body.append(paragraph)
    if current_title is not None:
        chapters.append(Chapter(book_number, len(chapters) + 1, TITLE_CORRECTIONS.get(current_title, current_title), "\n\n".join(current_body)))

    if expected != len(titles) or len(chapters) != len(titles):
        raise ValueError(f"Chapter parsing failed for {txt_path.name}: found {len(chapters)} of {len(titles)}")
    if any(not chapter.text for chapter in chapters):
        raise ValueError(f"Empty chapter found in {txt_path.name}")

    source = {
        "book_number": book_number,
        "text_file": txt_path.name,
        "text_sha256": sha256(txt_path),
        "chapter_count": len(chapters),
    }
    return chapters, source


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def split_oversized_sentence(tokenizer, prefix: str, sentence: str, max_tokens: int) -> List[str]:
    words = sentence.split()
    parts: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and token_count(tokenizer, prefix + "\n" + candidate) > max_tokens:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    if any(token_count(tokenizer, prefix + "\n" + part) > max_tokens for part in parts):
        raise ValueError("A single tokenized word exceeds the configured chunk limit")
    return parts


def sentence_parts(nlp, tokenizer, chapter: Chapter, max_tokens: int) -> List[str]:
    prefix = f"Book {chapter.book_number} | Chapter {chapter.chapter_number}: {chapter.title}"
    sentences = [sentence.text.strip() for sentence in nlp(chapter.text).sents if sentence.text.strip()]
    output: List[str] = []
    for sentence in sentences:
        if token_count(tokenizer, prefix + "\n" + sentence) <= max_tokens:
            output.append(sentence)
        else:
            output.extend(split_oversized_sentence(tokenizer, prefix, sentence, max_tokens))
    return output


def chunk_chapter(
    nlp,
    tokenizer,
    chapter: Chapter,
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
    overlap_sentences: int,
) -> Tuple[List[str], List[ChunkInfo]]:
    prefix = f"Book {chapter.book_number} | Chapter {chapter.chapter_number}: {chapter.title}"
    sentences = sentence_parts(nlp, tokenizer, chapter, max_tokens)
    groups: List[List[str]] = []
    current: List[str] = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and token_count(tokenizer, prefix + "\n" + candidate) > target_tokens:
            groups.append(current)
            overlap: List[str] = []
            for previous in current[-overlap_sentences:]:
                proposed = " ".join(overlap + [previous, sentence])
                if token_count(tokenizer, prefix + "\n" + proposed) <= target_tokens:
                    overlap.append(previous)
            current = overlap + [sentence]
        else:
            current.append(sentence)
    if current:
        groups.append(current)

    if len(groups) > 1:
        last_text = prefix + "\n" + " ".join(groups[-1])
        merged_text = prefix + "\n" + " ".join(groups[-2] + groups[-1])
        if token_count(tokenizer, last_text) < min_tokens and token_count(tokenizer, merged_text) <= max_tokens:
            groups[-2].extend(groups.pop())

    chunks: List[str] = []
    infos: List[ChunkInfo] = []
    for index, group in enumerate(groups, 1):
        text = prefix + "\n" + " ".join(group)
        size = token_count(tokenizer, text)
        if size > max_tokens:
            raise ValueError(f"Chunk exceeds {max_tokens} tokens: {prefix}, chunk {index}, got {size}")
        chunks.append(text)
        infos.append(ChunkInfo(chapter.book_number, chapter.chapter_number, chapter.title, index, size))
    return chunks, infos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--target-tokens", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=360)
    parser.add_argument("--min-tokens", type=int, default=80)
    parser.add_argument("--overlap-sentences", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.min_tokens < args.target_tokens <= args.max_tokens <= 384:
        raise ValueError("Require 0 < min_tokens < target_tokens <= max_tokens <= 384")
    reports = sorted(args.input.glob("Book *.report.json"), key=lambda path: int(re.match(r"Book\s+(\d+)", path.name).group(1)))
    if len(reports) != 7:
        raise ValueError(f"Expected seven cleanup reports, found {len(reports)}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.embedding_model), local_files_only=True)
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 10_000_000

    chapters: List[Chapter] = []
    sources: List[Dict[str, object]] = []
    for report in reports:
        book_chapters, source = parse_book(report)
        chapters.extend(book_chapters)
        sources.append(source)

    chunks: List[str] = []
    infos: List[ChunkInfo] = []
    for chapter in chapters:
        chapter_chunks, chapter_infos = chunk_chapter(
            nlp,
            tokenizer,
            chapter,
            args.target_tokens,
            args.max_tokens,
            args.min_tokens,
            args.overlap_sentences,
        )
        chunks.extend(chapter_chunks)
        infos.extend(chapter_infos)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format": {"chunks.json": "JSON array of strings"},
        "embedding_model": str(args.embedding_model.resolve()),
        "target_tokens": args.target_tokens,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
        "overlap_sentences": args.overlap_sentences,
        "book_count": len(sources),
        "chapter_count": len(chapters),
        "chunk_count": len(chunks),
        "min_chunk_tokens": min(info.token_count for info in infos),
        "max_chunk_tokens": max(info.token_count for info in infos),
        "average_chunk_tokens": sum(info.token_count for info in infos) / len(infos),
        "sources": sources,
        "chunks": [asdict(info) for info in infos],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Books: {len(sources)}")
    print(f"Chapters: {len(chapters)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunk tokens: min={manifest['min_chunk_tokens']} max={manifest['max_chunk_tokens']} avg={manifest['average_chunk_tokens']:.1f}")
    print(f"Wrote: {args.output / 'chunks.json'}")
    print(f"Wrote: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
