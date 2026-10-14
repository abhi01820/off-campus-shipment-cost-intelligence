"""In-memory vector retrieval engine for context notes with offline support and TF-IDF fallback."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# 1. Silence external library logs and progress bars
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for _noisy in ["httpx", "sentence_transformers", "transformers", "huggingface_hub", "urllib3"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# 2. Automatically detect if all-MiniLM-L6-v2 is cached and set HF_HUB_OFFLINE=1
_cache_dirs = [
    Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / ".cache" / "huggingface" / "hub",
]
for _cd in _cache_dirs:
    if _cd.exists() and any(_cd.glob("*all-MiniLM-L6-v2*")):
        os.environ["HF_HUB_OFFLINE"] = "1"
        break

logger = logging.getLogger(__name__)

# Global cached model to avoid re-instantiation across runs
_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load and cache the sentence-transformers model with graceful fallback."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # When offline and cached, local_files_only prevents external network checks
            is_offline = os.environ.get("HF_HUB_OFFLINE") == "1"
            _EMBEDDING_MODEL = SentenceTransformer(model_name, local_files_only=is_offline)
        except Exception as e:
            logger.info("SentenceTransformer not available or offline (%s). Using robust TF-IDF fallback.", e)
            _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


class NoteRetriever:
    """
    In-memory semantic vector search index over context notes.
    
    Supports:
    - Primary: sentence-transformers dense embeddings (all-MiniLM-L6-v2)
    - Fallback: scikit-learn TF-IDF vectorizer (robust, offline, deterministic)
    """

    def __init__(
        self,
        notes_df: pd.DataFrame,
        model_name: str = "all-MiniLM-L6-v2",
        force_tfidf: bool = False
    ):
        self.notes_df = notes_df.copy().reset_index(drop=True)
        self.model_name = model_name
        self.force_tfidf = force_tfidf
        self.model = None if force_tfidf else get_embedding_model(model_name)
        
        # Build text corpus for each note combining metadata and text
        self.documents: List[str] = []
        for _, row in self.notes_df.iterrows():
            doc = (
                f"Note ID: {row['note_id']} | Date: {pd.to_datetime(row['date']).strftime('%Y-%m-%d')} | "
                f"Route: {row['applies_to']} | Description: {row['note']}"
            )
            self.documents.append(doc)
            
        self.embeddings: np.ndarray = self._embed_documents(self.documents)

    def _embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.model is not None and not self.force_tfidf:
            emb = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return emb
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer().fit(texts)
            mat = self.tfidf.transform(texts).toarray()
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms

    def embed_query(self, query: str) -> np.ndarray:
        if self.model is not None and not self.force_tfidf:
            emb = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            return emb
        else:
            mat = self.tfidf.transform([query]).toarray()
            norm = np.linalg.norm(mat)
            return mat[0] / (norm if norm > 0 else 1.0)

    def retrieve(
        self,
        route: str,
        week_of: str,
        pct_vs_own: Optional[float] = None,
        pct_vs_peers: Optional[float] = None,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Query the index with route, week date, and cost-rise context.
        Returns top-k note records with similarity scores.
        """
        query_context = f"Route {route} experienced a freight cost increase during week {week_of}."
        if pct_vs_own and pct_vs_own > 0:
            query_context += f" Costs were up {pct_vs_own:.1f}% vs history."
        if pct_vs_peers and pct_vs_peers > 0:
            query_context += f" Costs were up {pct_vs_peers:.1f}% vs peers."

        query_emb = self.embed_query(query_context)
        # Cosine similarity (since both embeddings are L2 normalized)
        scores = np.dot(self.embeddings, query_emb)
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            sim = float(scores[idx])
            if sim < min_similarity:
                continue
            row = self.notes_df.iloc[idx].to_dict()
            row["similarity"] = sim
            results.append(row)

        return results
