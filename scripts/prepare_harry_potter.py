"""Extract, clean, and chunk a Harry Potter PDF for LinearRAG."""

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import spacy
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "Book" / "Book 1 - The Philosopher's Stone.pdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "harrypotter" / "chunks.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "dataset" / "harrypotter" / "manifest.json"

PAGE_HEADER_RE = re.compile(r"^P\s*a\s*g\s*e\s*\|\s*\d+.*\bRowling\s*$", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
HEADING_CORRECTIONS = {
    "THE VANASHIG GLASS": "THE VANISHING GLASS",
    "NICHOLAS FLAMEL": "NICOLAS FLAMEL",
}


@dataclass
class Chapter:
    number: int
    title: str
    start_page: int
    end_page: int
    text: str


@dataclass
class ChunkRecord:
    chapter_number: int
    chapter_title: str
    chunk_in_chapter: int
    word_count: int
    text: str


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    return re.sub(r"\s+", " ", line).strip()


def is_uppercase_heading(line: str) -> bool:
    letters = "".join(character for character in line if character.isalpha())
    return bool(letters) and letters == letters.upper() and 4 <= len(line) <= 80


def split_heading(lines: Sequence[str]) -> Tuple[Optional[str], List[str]]:
    heading_lines: List[str] = []
    for line in lines[:3]:
        if is_uppercase_heading(line) and not line.startswith(("“", '"', "—")):
            heading_lines.append(line)
        else:
            break

    if not heading_lines:
        return None, list(lines)

    heading = " ".join(heading_lines)
    heading = HEADING_CORRECTIONS.get(heading, heading)
    return heading, list(lines[len(heading_lines) :])


def join_wrapped_lines(lines: Iterable[str]) -> str:
    output = ""
    for line in lines:
        if not line:
            continue
        if output.endswith("-") and line[0].islower():
            output = output[:-1] + line
        else:
            output = f"{output} {line}" if output else line
    output = re.sub(r"\s+([,.;:!?])", r"\1", output)
    return re.sub(r"\s+", " ", output).strip()


def extract_chapters(pdf_path: Path) -> Tuple[List[Chapter], int, int]:
    reader = PdfReader(str(pdf_path))
    chapters: List[Chapter] = []
    current_title: Optional[str] = None
    current_start_page = 0
    current_pages: List[str] = []
    extracted_characters = 0

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        extracted_characters += len(raw_text)
        lines = [normalize_line(line) for line in raw_text.splitlines()]
        lines = [line for line in lines if line]

        if lines and PAGE_HEADER_RE.match(lines[0]):
            lines = lines[1:]

        heading, body_lines = split_heading(lines)
        if heading:
            if current_title is not None:
                chapters.append(
                    Chapter(
                        number=len(chapters) + 1,
                        title=current_title,
                        start_page=current_start_page,
                        end_page=page_number - 1,
                        text=join_wrapped_lines(current_pages),
                    )
                )
            current_title = heading
            current_start_page = page_number
            current_pages = []

        if current_title is not None:
            page_text = join_wrapped_lines(body_lines)
            if page_text:
                current_pages.append(page_text)

    if current_title is not None:
        chapters.append(
            Chapter(
                number=len(chapters) + 1,
                title=current_title,
                start_page=current_start_page,
                end_page=len(reader.pages),
                text=join_wrapped_lines(current_pages),
            )
        )

    if not chapters:
        raise ValueError("No chapter headings were detected in the PDF")

    return chapters, len(reader.pages), extracted_characters


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_long_sentence(sentence: str, max_words: int) -> List[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def chunk_chapter(
    chapter: Chapter,
    nlp,
    target_words: int,
    min_words: int,
    overlap_sentences: int,
) -> List[ChunkRecord]:
    sentences = [
        part
        for sentence in nlp(chapter.text).sents
        if sentence.text.strip()
        for part in split_long_sentence(sentence.text.strip(), target_words)
    ]
    core_chunks: List[List[str]] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = count_words(sentence)
        if current and current_words + sentence_words > target_words:
            core_chunks.append(current)
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words

    if current:
        core_chunks.append(current)

    if len(core_chunks) > 1 and sum(count_words(item) for item in core_chunks[-1]) < min_words:
        core_chunks[-2].extend(core_chunks.pop())

    records: List[ChunkRecord] = []
    for chunk_index, core_sentences in enumerate(core_chunks):
        sentence_group = list(core_sentences)
        if chunk_index and overlap_sentences:
            overlap: List[str] = []
            overlap_budget = target_words + min_words - sum(count_words(item) for item in core_sentences)
            for candidate in reversed(core_chunks[chunk_index - 1][-overlap_sentences:]):
                candidate_words = count_words(candidate)
                if candidate_words > overlap_budget:
                    break
                overlap.insert(0, candidate)
                overlap_budget -= candidate_words
            sentence_group = overlap + sentence_group
        body = " ".join(sentence_group).strip()
        prefix = f"Book 1 | Chapter {chapter.number}: {chapter.title}\n"
        records.append(
            ChunkRecord(
                chapter_number=chapter.number,
                chapter_title=chapter.title,
                chunk_in_chapter=chunk_index + 1,
                word_count=count_words(body),
                text=prefix + body,
            )
        )
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-words", type=int, default=180)
    parser.add_argument("--min-words", type=int, default=60)
    parser.add_argument("--overlap-sentences", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_words < 50:
        raise ValueError("--target-words must be at least 50")
    if args.min_words < 1 or args.min_words >= args.target_words:
        raise ValueError("--min-words must be positive and smaller than --target-words")
    if args.overlap_sentences < 0:
        raise ValueError("--overlap-sentences cannot be negative")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    chapters, page_count, extracted_characters = extract_chapters(args.input)
    sentence_nlp = spacy.blank("en")
    sentence_nlp.add_pipe("sentencizer")
    sentence_nlp.max_length = max(sentence_nlp.max_length, extracted_characters + 1)

    records = [
        record
        for chapter in chapters
        for record in chunk_chapter(
            chapter,
            sentence_nlp,
            target_words=args.target_words,
            min_words=args.min_words,
            overlap_sentences=args.overlap_sentences,
        )
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([record.text for record in records], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source_file": args.input.name,
        "source_sha256": sha256_file(args.input),
        "page_count": page_count,
        "extracted_characters": extracted_characters,
        "chapter_count": len(chapters),
        "chunk_count": len(records),
        "target_words": args.target_words,
        "min_words": args.min_words,
        "overlap_sentences": args.overlap_sentences,
        "chapters": [asdict(chapter) | {"text": None} for chapter in chapters],
        "chunks": [asdict(record) | {"text": None} for record in records],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    word_counts = [record.word_count for record in records]
    print(f"Source pages: {page_count}")
    print(f"Detected chapters: {len(chapters)}")
    print(f"Generated chunks: {len(records)}")
    print(f"Chunk words: min={min(word_counts)} max={max(word_counts)} avg={sum(word_counts) / len(word_counts):.1f}")
    print(f"Wrote: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
