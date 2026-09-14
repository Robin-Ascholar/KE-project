import json
import tempfile
import unittest
from pathlib import Path

from src.entity_alias import EntityAliasResolver, normalize_entity_name


class EntityAliasResolverTest(unittest.TestCase):
    def test_name_normalization(self):
        self.assertEqual(normalize_entity_name("The Dark-Lord's"), "dark lord")
        self.assertEqual(normalize_entity_name("YOU-KNOW-WHO"), "you know who")

    def test_alias_expands_to_available_graph_names(self):
        alias_path = Path("dataset/harrypotter/entity_aliases.json")
        resolver = EntityAliasResolver(str(alias_path))
        candidates = resolver.candidate_names(
            "Tom Riddle",
            ["Voldemort", "Tom Riddle", "Lord Voldemort", "Harry"],
        )
        names = [name for name, _ in candidates]
        self.assertEqual(names[0], "Tom Riddle")
        self.assertIn("Voldemort", names)
        self.assertIn("Lord Voldemort", names)
        scores = dict(candidates)
        self.assertEqual(scores["Tom Riddle"], 1.0)
        self.assertAlmostEqual(sum(scores[name] for name in names[1:]), 0.75)

    def test_duplicate_surface_variants_do_not_multiply_activation_mass(self):
        resolver = EntityAliasResolver(
            "dataset/harrypotter/entity_aliases.json",
            expansion_budget=0.6,
        )
        candidates = resolver.candidate_names(
            "Harry Potter",
            ["Harry Potter", "HARRY POTTER", "Harry", "HARRY", "The Boy Who Lived"],
        )
        names = [name for name, _ in candidates]
        self.assertEqual(names[0], "Harry Potter")
        self.assertEqual(len(names), 3)
        self.assertAlmostEqual(sum(score for _, score in candidates[1:]), 0.6)

    def test_alias_expansion_limit_is_applied_after_primary_seed(self):
        resolver = EntityAliasResolver(
            "dataset/harrypotter/entity_aliases.json",
            expansion_budget=0.5,
            max_expansions=2,
        )
        candidates = resolver.candidate_names(
            "Tom Riddle",
            ["Tom Riddle", "Voldemort", "Lord Voldemort", "Dark Lord"],
        )
        self.assertEqual(len(candidates), 3)
        self.assertAlmostEqual(sum(score for _, score in candidates[1:]), 0.5)

    def test_unknown_mention_keeps_original_path_available(self):
        resolver = EntityAliasResolver("dataset/harrypotter/entity_aliases.json")
        self.assertEqual(resolver.candidate_names("unknown person", ["Harry"]), [])

    def test_find_mentions_catches_aliases_without_ner(self):
        resolver = EntityAliasResolver("dataset/harrypotter/entity_aliases.json")
        mentions = resolver.find_mentions("Why did the Dark Lord fear the Boy Who Lived?")
        self.assertIn("Dark Lord", mentions)
        self.assertIn("The Boy Who Lived", mentions)
        self.assertEqual(resolver.find_mentions("Did Harrington meet Harry?"), ["Harry"])

    def test_ambiguous_alias_is_not_activated(self):
        payload = {
            "entities": [
                {"canonical_name": "A", "aliases": ["shared"]},
                {"canonical_name": "B", "aliases": ["shared"]},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aliases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            resolver = EntityAliasResolver(str(path))
            self.assertIsNone(resolver.canonical_for("shared"))
            self.assertIn("shared", resolver.ambiguous_aliases)


if __name__ == "__main__":
    unittest.main()
