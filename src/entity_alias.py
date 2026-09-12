"""Alias-aware entity resolution for query-time entity activation.

The resolver keeps the existing graph entities unchanged. It only adds a
logical alias layer at query time, so existing indexes remain compatible.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AliasEntity:
    """One logical entity and its surface forms."""

    canonical_name: str
    aliases: Tuple[str, ...]
    entity_id: str

    @property
    def names(self) -> Tuple[str, ...]:
        return (self.canonical_name,) + self.aliases


def normalize_entity_name(value: object) -> str:
    """Normalize a surface form for exact alias lookup.

    This is deliberately conservative: it handles case, Unicode punctuation,
    possessives, hyphens, and a leading article without doing fuzzy matching.
    """

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    tokens = text.split()
    if tokens and tokens[0] in {"a", "an", "the"}:
        tokens = tokens[1:]
    return " ".join(tokens)


class EntityAliasResolver:
    """Resolve query mentions to available graph entity surface forms.

    The graph still stores its original entity strings. For a configured alias
    group, all group members that exist in the current entity index are
    returned as activation candidates.
    """

    def __init__(
        self,
        alias_path: Optional[str] = None,
        *,
        enabled: bool = True,
        activation_score: float = 0.85,
    ) -> None:
        self.enabled = enabled
        self.activation_score = float(activation_score)
        self.alias_path = str(alias_path) if alias_path else None
        self.entities: Dict[str, AliasEntity] = {}
        self.alias_to_canonical: Dict[str, str] = {}
        self.normalized_to_surface: Dict[str, str] = {}
        self.ambiguous_aliases: Tuple[str, ...] = ()
        if self.enabled and alias_path:
            self._load(Path(alias_path))

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.entities)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_entities = payload.get("entities", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_entities, list):
            raise ValueError(f"Alias file must contain an entities list: {path}")

        pending: Dict[str, AliasEntity] = {}
        normalized_owners: Dict[str, str] = {}
        normalized_surfaces: Dict[str, str] = {}
        conflicts = set()
        for raw in raw_entities:
            if not isinstance(raw, dict):
                raise ValueError(f"Each alias entity must be an object: {path}")
            canonical = str(raw.get("canonical_name", "")).strip()
            if not canonical:
                raise ValueError(f"Alias entity is missing canonical_name: {path}")
            aliases = raw.get("aliases", [])
            if not isinstance(aliases, list):
                raise ValueError(f"aliases must be a list for {canonical}: {path}")

            names: List[str] = []
            for name in [canonical] + aliases:
                name = str(name).strip()
                if name and name not in names:
                    names.append(name)
            entity = AliasEntity(
                canonical_name=canonical,
                aliases=tuple(name for name in names if name != canonical),
                entity_id=self._entity_id(canonical),
            )
            canonical_key = normalize_entity_name(canonical)
            if canonical_key in pending:
                raise ValueError(f"Duplicate canonical_name: {canonical}")
            pending[canonical_key] = entity

            for name in entity.names:
                key = normalize_entity_name(name)
                if not key:
                    continue
                owner = normalized_owners.get(key)
                if owner is not None and owner != canonical_key:
                    conflicts.add(key)
                else:
                    normalized_owners[key] = canonical_key
                    normalized_surfaces.setdefault(key, name)

        for key in conflicts:
            normalized_owners.pop(key, None)
            normalized_surfaces.pop(key, None)
        self.entities = pending
        self.alias_to_canonical = normalized_owners
        self.normalized_to_surface = normalized_surfaces
        self.ambiguous_aliases = tuple(sorted(conflicts))

    @staticmethod
    def _entity_id(canonical_name: str) -> str:
        digest = hashlib.md5(normalize_entity_name(canonical_name).encode("utf-8")).hexdigest()
        return f"alias-{digest}"

    def canonical_for(self, mention: object) -> Optional[str]:
        if not self.available:
            return None
        canonical_key = self.alias_to_canonical.get(normalize_entity_name(mention))
        entity = self.entities.get(canonical_key) if canonical_key else None
        return entity.canonical_name if entity else None

    def entity_for(self, mention: object) -> Optional[AliasEntity]:
        if not self.available:
            return None
        canonical_key = self.alias_to_canonical.get(normalize_entity_name(mention))
        return self.entities.get(canonical_key) if canonical_key else None

    def candidate_names(
        self,
        mention: object,
        available_entities: Sequence[str],
    ) -> List[Tuple[str, float]]:
        """Return available graph names and their alias activation scores."""

        entity = self.entity_for(mention)
        if entity is None:
            return []

        available_by_key: Dict[str, List[str]] = {}
        for name in available_entities:
            key = normalize_entity_name(name)
            if key:
                available_by_key.setdefault(key, []).append(name)

        mention_key = normalize_entity_name(mention)
        ordered_names = sorted(
            entity.names,
            key=lambda name: (
                0 if normalize_entity_name(name) == mention_key else 1,
                0 if normalize_entity_name(name) == normalize_entity_name(entity.canonical_name) else 1,
                normalize_entity_name(name),
            ),
        )
        candidates: List[Tuple[str, float]] = []
        seen = set()
        for name in ordered_names:
            key = normalize_entity_name(name)
            for available_name in available_by_key.get(key, []):
                if available_name in seen:
                    continue
                seen.add(available_name)
                score = 1.0 if key == mention_key else self.activation_score
                candidates.append((available_name, score))
        return candidates

    def find_mentions(self, text: object) -> List[str]:
        """Find configured alias phrases in free-form query text.

        This supplements NER for aliases such as ``Dark Lord`` that may not
        be tagged as entities by the configured spaCy model. Matching is
        normalized and token-bound, with longer phrases taking precedence over
        shorter overlapping forms.
        """

        if not self.available:
            return []
        normalized_text = normalize_entity_name(text)
        if not normalized_text:
            return []

        matches = []
        keys = sorted(
            self.alias_to_canonical,
            key=lambda key: (-len(key.split()), -len(key), key),
        )
        occupied_spans: List[Tuple[int, int]] = []
        for key in keys:
            pattern = rf"(?<!\w){re.escape(key)}(?!\w)"
            for match in re.finditer(pattern, normalized_text):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in occupied_spans):
                    continue
                occupied_spans.append(span)
                matches.append((span[0], -len(key), self.normalized_to_surface[key]))

        matches.sort(key=lambda item: (item[0], item[1], item[2].casefold()))
        return [surface for _, _, surface in matches]

    def expansion_summary(
        self,
        mention: object,
        available_entities: Sequence[str],
    ) -> Optional[Dict[str, object]]:
        entity = self.entity_for(mention)
        if entity is None:
            return None
        candidates = self.candidate_names(mention, available_entities)
        return {
            "mention": str(mention),
            "canonical_name": entity.canonical_name,
            "canonical_entity_id": entity.entity_id,
            "activated_entities": [name for name, _ in candidates],
        }

    def validate_available_entities(self, available_entities: Iterable[str]) -> Dict[str, object]:
        """Return coverage information for an existing entity index."""

        available = list(available_entities)
        covered = {}
        for entity in self.entities.values():
            names = [name for name, _ in self.candidate_names(entity.canonical_name, available)]
            covered[entity.canonical_name] = names
        return {
            "alias_group_count": len(self.entities),
            "available_entity_count": len(available),
            "groups_with_available_names": sum(bool(names) for names in covered.values()),
            "coverage": covered,
            "ambiguous_aliases": list(self.ambiguous_aliases),
        }
