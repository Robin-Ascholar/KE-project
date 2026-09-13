"""Utilities for book-aware soft routing in long-series corpora.

The router is deliberately embedding-only: it estimates a distribution over
books from the existing question-to-passage similarities and never calls an
LLM.  Keeping the math here makes the behavior independently testable and
keeps the default LinearRAG path unchanged when routing is disabled.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np


BOOK_PREFIX_RE = re.compile(r"^(?:\d+:)?Book\s+(\d+)\s*\|", re.IGNORECASE)


def extract_book_number(passage: str) -> int | None:
    """Return the book number encoded in a prepared passage prefix."""

    match = BOOK_PREFIX_RE.match(str(passage).strip())
    return int(match.group(1)) if match else None


def normalized_entropy_confidence(probabilities: Sequence[float]) -> float:
    """Map distribution entropy to confidence in [0, 1]."""

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or len(probs) == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if len(probs) == 1:
        return 1.0
    total = float(probs.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("probabilities must have a positive finite sum")
    probs = probs / total
    positive = probs[probs > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    confidence = 1.0 - entropy / math.log(len(probs))
    return float(np.clip(confidence, 0.0, 1.0))


def infer_book_distribution(
    passage_similarities: Sequence[float],
    passage_books: Sequence[int],
    *,
    book_ids: Iterable[int] | None = None,
    top_m: int = 3,
    temperature: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer P(book | question) using the mean of each book's top-m passages.

    Top-m mean pooling avoids favoring longer books merely because they contain
    more passages.  The returned tuple is ``(book_ids, book_scores, probs)``.
    """

    similarities = np.asarray(passage_similarities, dtype=float)
    books = np.asarray(passage_books, dtype=int)
    if similarities.ndim != 1 or books.ndim != 1 or len(similarities) != len(books):
        raise ValueError("passage_similarities and passage_books must be aligned 1-D arrays")
    if len(similarities) == 0:
        raise ValueError("at least one passage is required")
    if top_m < 1:
        raise ValueError("top_m must be positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")

    ids = np.asarray(
        sorted(set(int(value) for value in (book_ids if book_ids is not None else books) if int(value) > 0)),
        dtype=int,
    )
    if len(ids) == 0:
        raise ValueError("no positive book IDs were provided")

    scores = []
    for book_id in ids:
        values = similarities[books == book_id]
        if len(values) == 0:
            raise ValueError(f"book {book_id} has no passages")
        count = min(top_m, len(values))
        top_values = np.partition(values, len(values) - count)[-count:]
        scores.append(float(np.mean(top_values)))
    score_array = np.asarray(scores, dtype=float)

    logits = (score_array - float(np.max(score_array))) / temperature
    exp_logits = np.exp(logits)
    probabilities = exp_logits / float(np.sum(exp_logits))
    return ids, score_array, probabilities


def book_prior_factors(
    passage_books: Sequence[int],
    book_ids: Sequence[int],
    probabilities: Sequence[float],
    *,
    strength: float,
    confidence: float = 1.0,
    boost_only: bool = False,
) -> tuple[np.ndarray, float]:
    """Return multiplicative passage factors and the effective strength.

    The default ``factor=(1-lambda)+lambda*B*P(book|q)`` makes a uniform
    distribution an exact no-op.  With ``boost_only=True``, factors are floored
    at 1 so likely books are promoted without suppressing cross-book evidence.
    """

    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")

    books = np.asarray(passage_books, dtype=int)
    ids = np.asarray(book_ids, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if ids.ndim != 1 or probs.ndim != 1 or len(ids) != len(probs) or len(ids) == 0:
        raise ValueError("book_ids and probabilities must be aligned non-empty arrays")
    if np.any(probs < 0) or not np.isfinite(probs).all() or float(probs.sum()) <= 0:
        raise ValueError("probabilities must be finite and non-negative")
    probs = probs / float(probs.sum())

    probability_by_book = {int(book_id): float(probability) for book_id, probability in zip(ids, probs)}
    unknown = sorted(set(int(book) for book in books) - set(probability_by_book))
    if unknown:
        raise ValueError(f"passages contain unknown book IDs: {unknown}")

    effective_strength = strength * confidence
    centered_priors = np.asarray(
        [len(ids) * probability_by_book[int(book)] - 1.0 for book in books],
        dtype=float,
    )
    if boost_only:
        centered_priors = np.maximum(centered_priors, 0.0)
    factors = 1.0 + effective_strength * centered_priors
    return factors, effective_strength
