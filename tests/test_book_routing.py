import unittest

import numpy as np

from src.book_routing import (
    book_prior_factors,
    extract_book_number,
    infer_book_distribution,
    normalized_entropy_confidence,
)


class BookRoutingTests(unittest.TestCase):
    def test_extract_book_number_accepts_indexed_and_plain_prefixes(self):
        self.assertEqual(extract_book_number("0:Book 1 | Chapter 1: Title\nText"), 1)
        self.assertEqual(extract_book_number("Book 7 | Chapter 37: Title"), 7)
        self.assertIsNone(extract_book_number("Chapter 1: no book prefix"))

    def test_top_m_mean_prefers_relevant_book_without_length_bias(self):
        similarities = np.asarray([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1])
        books = np.asarray([1, 1, 2, 2, 2, 2, 2])
        book_ids, scores, probabilities = infer_book_distribution(
            similarities,
            books,
            top_m=2,
            temperature=0.1,
        )
        self.assertEqual(book_ids.tolist(), [1, 2])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(probabilities[0], 0.99)

    def test_uniform_distribution_is_noop(self):
        factors, effective = book_prior_factors(
            [1, 2, 1, 2],
            [1, 2],
            [0.5, 0.5],
            strength=0.75,
        )
        np.testing.assert_allclose(factors, np.ones(4))
        self.assertEqual(effective, 0.75)

    def test_soft_prior_keeps_non_target_floor(self):
        factors, _ = book_prior_factors(
            [1, 2],
            [1, 2],
            [1.0, 0.0],
            strength=0.5,
        )
        np.testing.assert_allclose(factors, [1.5, 0.5])

    def test_boost_only_prior_never_suppresses_other_books(self):
        factors, effective = book_prior_factors(
            [1, 2],
            [1, 2],
            [1.0, 0.0],
            strength=0.5,
            boost_only=True,
        )
        np.testing.assert_allclose(factors, [1.5, 1.0])
        self.assertTrue(np.all(factors >= 1.0))
        self.assertEqual(effective, 0.5)

    def test_entropy_confidence_is_low_for_ambiguous_queries(self):
        self.assertAlmostEqual(normalized_entropy_confidence([0.5, 0.5]), 0.0)
        self.assertAlmostEqual(normalized_entropy_confidence([1.0, 0.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
