"""
vector_store/faiss_store.py

WHAT IS FAISS?
FAISS (Facebook AI Similarity Search) is a library that answers one question:
  "Given a query vector, which of these 18,000 stored vectors is most similar?"

Naively you'd compute similarity against every stored vector — that's O(N).
FAISS uses indexing structures to do this in sub-linear time.

WHICH INDEX TYPE?
We use IndexHNSWFlat (Hierarchical Navigable Small World graph).

How it works (simple version):
  Build a graph where each vector is a node.
  Nearby vectors are connected by edges.
  Search = start at a random node, greedily hop toward the query.

┌─────────────────┬──────────────────────────────────────────┐
│ Index Type      │ Tradeoff                                 │
├─────────────────┼──────────────────────────────────────────┤
│ IndexFlatIP     │ Exact search. Slow at scale. Good for <1k│
│ IndexHNSWFlat   │ ~Exact. Fast. Our choice for 18k docs.   │
│ IndexIVFFlat    │ Faster but needs training. Overkill here.│
└─────────────────┴──────────────────────────────────────────┘

WHY NOT IndexFlatL2?
We use Inner Product (IP) not L2 because our vectors are normalized.
For normalized vectors: cosine_similarity = inner_product.
This is both mathematically correct and faster.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Stores document embeddings in a FAISS HNSW index.
    Supports adding documents, searching by vector, and filtering by cluster.
    """

    def __init__(self, dimension: int = 384, hnsw_m: int = 32):
        """
        Args:
            dimension: Size of each embedding vector (384 for MiniLM).
            hnsw_m: Number of connections per node in the HNSW graph.
                    Higher M = better recall, more memory.
                    32 is a standard production-grade value.

        HNSW Parameters explained:
          M=32: Each node connects to 32 neighbours. More = better search quality.
          efConstruction=200: How many candidates to explore while building the graph.
                              Higher = better quality index, slower build time.
          efSearch (set at query time): How many candidates to explore during search.
        """
        self.dimension = dimension
        self.hnsw_m = hnsw_m

        # IndexHNSWFlat with Inner Product similarity
        # IP = Inner Product. Since our vectors are normalized, IP = cosine similarity.
        self.index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200

        # Metadata storage (parallel array — index i corresponds to faiss internal id i)
        # FAISS stores vectors but NOT metadata. We keep metadata separately.
        self.metadata: list[dict] = []

        logger.info(f"FAISS HNSW index initialized (dim={dimension}, M={hnsw_m})")

    def add_documents(
        self,
        embeddings: np.ndarray,
        metadata_list: list[dict],
    ) -> None:
        """
        Add document vectors to the index.

        Args:
            embeddings: Shape (N, dimension). Must be float32.
            metadata_list: List of dicts, one per document.
                          Each dict should contain at least:
                          {'doc_id', 'text', 'category', 'cluster_memberships'}

        NOTE: FAISS assigns sequential integer IDs (0, 1, 2, ...).
        We keep `self.metadata` aligned so metadata[i] describes the vector at index i.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) != metadata count ({len(metadata_list)})"
            )

        # FAISS requires float32 specifically
        embeddings_f32 = embeddings.astype(np.float32)

        logger.info(f"Adding {len(embeddings_f32)} vectors to FAISS index...")
        self.index.add(embeddings_f32)
        self.metadata.extend(metadata_list)

        logger.info(f"Index now contains {self.index.ntotal} vectors")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        cluster_filter: int | None = None,
    ) -> list[dict]:
        """
        Find the top_k most similar documents to a query vector.

        Args:
            query_vector: 1D array of shape (dimension,). Must be normalized.
            top_k: How many results to return.
            cluster_filter: If set, only return docs from this cluster.
                           (We over-fetch and then filter — see note below.)

        Returns:
            List of result dicts, sorted by similarity descending.
            Each dict: {'doc_id', 'similarity', 'text', 'category', ...}

        NOTE ON CLUSTER FILTERING:
        FAISS doesn't natively support metadata filtering.
        Our approach: fetch top_k * 5 results, then filter by cluster.
        This is called "over-fetch and post-filter" — simple and effective
        for our scale. At millions of docs you'd use a proper filtered index.
        """
        # Reshape to (1, dimension) — FAISS expects batched queries
        query_f32 = query_vector.astype(np.float32).reshape(1, -1)

        # Set efSearch — higher = more accurate but slower search
        self.index.hnsw.efSearch = 128

        # How many to fetch before filtering
        fetch_k = top_k * 5 if cluster_filter is not None else top_k

        # FAISS search returns:
        #   similarities: shape (1, fetch_k) — the inner product scores
        #   indices: shape (1, fetch_k)      — the FAISS internal IDs
        similarities, indices = self.index.search(query_f32, min(fetch_k, self.index.ntotal))

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for padding when not enough results exist
                continue

            meta = self.metadata[idx].copy()
            meta['similarity'] = float(sim)

            # Post-filter: skip if cluster doesn't match
            if cluster_filter is not None:
                dominant = meta.get('dominant_cluster')
                if dominant != cluster_filter:
                    continue

            results.append(meta)

            if len(results) >= top_k:
                break

        return results

    def get_all_embeddings(self) -> np.ndarray:
        """
        Reconstruct all stored vectors from the index.
        Used during clustering — we need the raw vectors for FCM.

        Note: This only works with IndexHNSWFlat (flat = stores raw vectors).
        Quantized indexes (IVF, PQ) would require different handling.
        """
        n = self.index.ntotal
        # Allocate output buffer
        embeddings = np.zeros((n, self.dimension), dtype=np.float32)
        # FAISS reconstruct_n: copies vectors back out
        self.index.storage.reconstruct_n(0, n, embeddings)
        return embeddings

    def save(self, index_path: str, metadata_path: str) -> None:
        """Persist the FAISS index and metadata to disk."""
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved index to {index_path}")
        logger.info(f"Saved metadata to {metadata_path}")

    @classmethod
    def load(cls, index_path: str, metadata_path: str, dimension: int = 384) -> "FAISSVectorStore":
        """Load a previously saved vector store."""
        store = cls.__new__(cls)
        store.dimension = dimension
        store.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            store.metadata = pickle.load(f)
        logger.info(f"Loaded FAISS index: {store.index.ntotal} vectors")
        return store

    @property
    def size(self) -> int:
        return self.index.ntotal