from dataclasses import dataclass, field
from typing import Optional
from src.utils import LLM_Model
@dataclass
class LinearRAGConfig:
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False  # True for vectorized matrix computation, False for BFS iteration
    enable_chapter_retrieval: bool = False
    chapter_top_k: int = 8
    chapter_global_fallback: int = 2
    save_graph_index: bool = True
    entity_alias_path: Optional[str] = None
    enable_alias_activation: bool = True
    # Alias nodes share this budget per logical entity.  It is deliberately a
    # total budget rather than a per-node score, otherwise large alias groups
    # dominate Personalized PageRank merely because they contain more names.
    alias_expansion_budget: float = 0.75
    max_alias_expansions: int = 8
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])
