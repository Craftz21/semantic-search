"""
embeddings/embedder.py

WHAT IS AN EMBEDDING?
A sentence embedding is a list of ~384 numbers that represents the *meaning*
of a piece of text. The key property: texts with similar meanings end up
with similar number-lists (vectors).

Example:
  "gun control laws"      → [0.12, -0.34, 0.89, ...]
  "firearm legislation"   → [0.11, -0.31, 0.91, ...]  ← very close
  "space shuttle launch"  → [-0.67, 0.22, -0.44, ...] ← very different

We measure similarity using COSINE SIMILARITY:
  - Two identical texts → cosine similarity = 1.0
  - Two unrelated texts → cosine similarity ≈ 0.0
  - Opposite meanings   → cosine similarity = -1.0

WHY NORMALIZE?
After normalization (making every vector length = 1.0),
cosine similarity becomes identical to dot product.
This is important because FAISS's fast search uses dot product internally.
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Wraps SentenceTransformer to embed text and normalize outputs.

    The normalization step is crucial:
      raw_vector / ||raw_vector|| = unit vector
      ||unit_vector|| = 1.0

    After this, cosine_similarity(a, b) = dot_product(a, b)
    which lets FAISS use its fastest search mode (IndexFlatIP = Inner Product).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        WHY all-MiniLM-L6-v2?
        ┌──────────────────────┬────────────────────────────────────────────┐
        │ Property             │ Value                                      │
        ├──────────────────────┼────────────────────────────────────────────┤
        │ Embedding dimensions │ 384 (small = fast)                         │
        │ Max input tokens     │ 256 (long docs get truncated, we handle it)│
        │ Semantic quality     │ Very strong for similarity tasks           │
        │ Speed (CPU)          │ ~2000 sentences/second                     │
        │ Model size           │ ~22MB (downloads once, caches locally)     │
        └──────────────────────┴────────────────────────────────────────────┘

        Alternatives we considered:
        - all-mpnet-base-v2: Better quality but 768-dim (2x memory, 2x slower)
        - all-MiniLM-L12-v2: 12-layer version, marginally better, 2x slower
        - text-embedding-3-small (OpenAI): Requires API key + cost per call

        For 18,000 documents on a local machine, MiniLM-L6 is the right call.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed one piece of text. Returns a normalized 1D numpy array.
        Used at query time (fast, single inference).
        """
        vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return self._normalize(vec)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of texts efficiently using batching.

        WHY BATCHING?
        The model processes texts in parallel on each batch.
        batch_size=64 is a sweet spot for CPU — larger batches use more RAM
        but don't speed things up much beyond ~64 on CPU.
        On GPU you could push to 256+.

        Returns:
            Matrix of shape (len(texts), 384)
            Each row is the normalized embedding for one text.
        """
        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})...")

        all_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize manually below
        )

        # Normalize every vector to unit length
        normalized = self._normalize_matrix(all_embeddings)

        logger.info(f"Embedding complete. Shape: {normalized.shape}")
        return normalized

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """
        Divide a single vector by its L2 norm (Euclidean length).
        Result: a vector pointing in the same direction but with length = 1.0

        Why? So that cosine similarity = dot product.
        cos(θ) = (a · b) / (||a|| × ||b||)
        If ||a|| = ||b|| = 1, then cos(θ) = a · b
        """
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec  # Zero vector stays zero (shouldn't happen with real text)
        return vec / norm

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Normalize each row of a 2D matrix independently.
        matrix shape: (N, 384) → output shape: (N, 384)
        Each row becomes a unit vector.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Avoid division by zero for any all-zero rows
        norms = np.where(norms == 0, 1, norms)
        return matrix / norms

    def save_embeddings(self, embeddings: np.ndarray, path: str) -> None:
        """Save embeddings matrix to disk as a .npy binary file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        size_mb = embeddings.nbytes / (1024 * 1024)
        logger.info(f"Saved embeddings to {path} ({size_mb:.1f} MB)")

    def load_embeddings(self, path: str) -> np.ndarray:
        """Load previously saved embeddings."""
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings from {path}. Shape: {embeddings.shape}")
        return embeddings