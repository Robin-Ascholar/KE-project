import unittest

from src.entity_aliases import EntityAliasResolver, normalize_alias


GROUPS = {
    "Lord Voldemort": ["Voldemort", "Tom Riddle", "You-Know-Who"],
    "Peter Pettigrew": ["Wormtail", "Scabbers"],
}


class EntityAliasResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = EntityAliasResolver(GROUPS)

    def test_normalizes_punctuation_and_case(self):
        self.assertEqual(normalize_alias("You-Know-Who"), "you know who")

    def test_matches_alias_and_expands_canonical_group(self):
        primary, expansions, matched = self.resolver.expansion_plan(
            "What was Scabbers's true identity?", []
        )
        self.assertEqual(primary, ["Scabbers"])
        self.assertEqual(matched, {"Peter Pettigrew": ["Scabbers"]})
        self.assertEqual(expansions["Peter Pettigrew"], ["Peter Pettigrew", "Wormtail"])

    def test_does_not_match_partial_words(self):
        self.assertEqual(self.resolver.matched_groups("A riddled diary"), {})

    def test_does_not_merge_tom_riddle_senior(self):
        self.assertEqual(
            self.resolver.matched_groups("Where was Tom Riddle Senior buried?"),
            {},
        )


if __name__ == "__main__":
    unittest.main()
