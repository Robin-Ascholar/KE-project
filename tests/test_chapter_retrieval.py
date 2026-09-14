import unittest

import numpy as np

try:
    from src.LinearRAG import LinearRAG
    from src.config import LinearRAGConfig
    IMPORT_ERROR = None
except ModuleNotFoundError as error:
    LinearRAG = None
    LinearRAGConfig = None
    IMPORT_ERROR = error


@unittest.skipIf(LinearRAG is None, f"project dependencies unavailable: {IMPORT_ERROR}")
class ChapterRetrievalTests(unittest.TestCase):
    def test_experimental_features_are_opt_in(self):
        config = LinearRAGConfig(dataset_name="test")

        self.assertFalse(config.enable_chapter_retrieval)
        self.assertFalse(config.enable_alias_activation)
        self.assertEqual(config.chapter_top_k, 16)
        self.assertEqual(config.chapter_global_fallback, 4)

    def test_chapter_candidates_keep_global_fallback(self):
        rag = LinearRAG.__new__(LinearRAG)
        rag.chapter_keys = ["book1-chapter1", "book1-chapter2"]
        rag.chapter_texts = {
            "book1-chapter1": "Book 1 | Chapter 1: Start",
            "book1-chapter2": "Book 1 | Chapter 2: Middle",
        }
        rag.chapter_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        rag.passage_to_chapter = {0: "book1-chapter1", 1: "book1-chapter1", 2: "book1-chapter2"}
        rag.passage_embedding_store = type("Store", (), {"hash_ids": ["p0", "p1", "p2"]})()
        rag.passage_embeddings = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
        rag.config = type("Config", (), {"chapter_top_k": 1, "chapter_global_fallback": 1})()

        allowed, info = rag._chapter_candidates(np.array([1.0, 0.0]))

        self.assertEqual(info["selected_chapters"][0]["chapter"], "Book 1 | Chapter 1: Start")
        self.assertIn("p0", allowed)
        self.assertEqual(info["chapter_candidate_passages"], 2)

    def test_metadata_parser_groups_passages(self):
        rag = LinearRAG.__new__(LinearRAG)
        passages = [
            "0:Book 1 | Chapter 1: Start\nfirst",
            "1:Book 1 | Chapter 1: Start\nsecond",
            "2:Book 1 | Chapter 2: Middle\nthird",
        ]
        rag._load_chapter_metadata(passages)
        self.assertEqual(rag.chapter_keys, ["book1-chapter1", "book1-chapter2"])
        self.assertEqual(rag.passage_to_chapter[2], "book1-chapter2")


if __name__ == "__main__":
    unittest.main()
