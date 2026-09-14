#!/usr/bin/env python3
"""Validate alias groups and report their coverage in an existing entity index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.entity_alias import EntityAliasResolver  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aliases",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "harrypotter" / "entity_aliases.json",
    )
    parser.add_argument(
        "--ner-results",
        type=Path,
        default=PROJECT_ROOT / "import" / "harrypotter" / "ner_results.json",
        help="Optional NER result JSON used to measure available entity-name coverage",
    )
    return parser.parse_args()


def load_available_entities(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entity_names = set()
    for entities in payload.get("passage_hash_id_to_entities", {}).values():
        entity_names.update(str(entity) for entity in entities)
    for entities in payload.get("sentence_to_entities", {}).values():
        entity_names.update(str(entity) for entity in entities)
    return sorted(entity_names)


def main() -> None:
    args = parse_args()
    resolver = EntityAliasResolver(str(args.aliases))
    available_entities = load_available_entities(args.ner_results)
    report = resolver.validate_available_entities(available_entities)

    print(f"Alias groups: {report['alias_group_count']}")
    print(f"Available NER entities: {report['available_entity_count']}")
    print(f"Groups with at least one available name: {report['groups_with_available_names']}")
    if report["ambiguous_aliases"]:
        print("Ambiguous aliases skipped: " + ", ".join(report["ambiguous_aliases"]))

    missing = [
        canonical
        for canonical, names in report["coverage"].items()
        if not names
    ]
    if missing:
        print("Groups without an indexed surface form:")
        for canonical in missing:
            print(f"  - {canonical}")
    else:
        print("Every alias group has at least one indexed surface form.")


if __name__ == "__main__":
    main()
