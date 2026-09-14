"""Query-time entity alias expansion with conservative lexical matching."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence


def normalize_alias(value: str) -> str:
    """Normalize punctuation and whitespace for alias matching/deduplication."""

    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


class EntityAliasResolver:
    """Resolve surface names to curated canonical groups without fuzzy matching."""

    def __init__(self, groups: Mapping[str, Sequence[str]] | None = None):
        self.groups: dict[str, tuple[str, ...]] = {}
        for canonical, aliases in (groups or {}).items():
            names = []
            seen = set()
            for name in (canonical, *aliases):
                normalized = normalize_alias(name)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    names.append(str(name))
            if names:
                self.groups[str(canonical)] = tuple(names)

    @staticmethod
    def _surface_occurs(normalized_question: str, normalized_alias: str) -> bool:
        pattern = re.compile(rf"(?<!\w){re.escape(normalized_alias)}(?!\w)")
        matches = list(pattern.finditer(normalized_question))
        if normalized_alias != "tom riddle":
            return bool(matches)

        # Do not merge Voldemort with his Muggle father, Tom Riddle Sr./Senior.
        for match in matches:
            suffix = normalized_question[match.end():].lstrip()
            if not re.match(r"^(?:sr|senior)\b", suffix):
                return True
        return False

    def matched_groups(self, question: str) -> dict[str, list[str]]:
        """Return canonical groups and the surface aliases present in a question."""

        normalized_question = normalize_alias(question)
        matched: dict[str, list[str]] = {}
        for canonical, names in self.groups.items():
            surfaces = [
                name
                for name in names
                if self._surface_occurs(normalized_question, normalize_alias(name))
            ]
            if surfaces:
                matched[canonical] = surfaces
        return matched

    def expansion_plan(
        self,
        question: str,
        detected_entities: Sequence[str],
    ) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
        """Return primary terms, per-group expansion terms, and matched surfaces."""

        matched = self.matched_groups(question)
        primary: list[str] = []
        seen_primary = set()
        for term in (*detected_entities, *(name for names in matched.values() for name in names)):
            normalized = normalize_alias(term)
            if normalized and normalized not in seen_primary:
                seen_primary.add(normalized)
                primary.append(str(term))

        expansions: dict[str, list[str]] = {}
        for canonical in matched:
            terms = [
                name
                for name in self.groups[canonical]
                if normalize_alias(name) not in seen_primary
            ]
            if terms:
                expansions[canonical] = terms
        return primary, expansions, matched
