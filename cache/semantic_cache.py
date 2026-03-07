"""
cache/semantic_cache.py

THE PROBLEM WITH TRADITIONAL CACHES:
  Traditional: cache["gun control"] → result
  Fails for:   cache["firearm legislation"] → MISS (different string, same meaning)

OUR SOLUTION — SEMANTIC CACHE:
  Store (query_vector, result) pairs.
  On new query: compute similarity between new vector and all cached vectors.
  If similarity > threshold → HIT (same meaning, different words)

THE CLUSTER-AWARE IMPROVEMENT:
  Naive semantic cache searches ALL cached entries → O(N) as cache grows.
  Our cache organizes entries BY CLUSTER.
  A query routed to cluster 7 only searches cluster 7's cache entries.
  This reduces search from O(N_total) to O(N_cluster) — typically 10-20x faster.

  Why is this valid?
  Two queries that mean the same thing will have similar embeddings.
  Similar embeddings → same dominant cluster (by FCM).
  So similar queries will be routed to the same cluster cache.

ADAPTIVE THRESHOLD:
  Not all clusters deserve the same threshold.

  Tight cluster (e.g., sci.crypt): all docs use very specific crypto terminology.
    → Even slightly different queries may be semantically distinct.
    → Use HIGHER threshold (0.88) to avoid false positives.

  Loose cluster (e.g., talk.politics.misc): broad, diverse language.
    → Paraphrased queries may have lower cosine similarity.
    → Use LOWER threshold (0.78) to still catch paraphrases.

  Formula: threshold_c = base_threshold + adjustment(variance_c)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """
    Represents one cached query-result pair.

    We store:
    - The query vector (for similarity comparison)
    - The original query string (for the API response's 'matched_query' field)
    - The result (what we return on a cache hit)
    - Metadata for analysis
    """
    query_text: str
    query_vector: np.ndarray        # Shape: (384,) — normalized
    result: str                     # The computed result/answer
    dominant_cluster: int           # Which cluster this query belongs to
    cluster_memberships: np.ndarray # Full membership distribution (K,)
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0              # How many times this entry was a cache hit

    def record_hit(self):
        self.hit_count += 1


# ── The Cache ─────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Cluster-aware semantic cache built from scratch.
    No Redis, Memcached, or any caching library.

    Internal structure:
        _store: dict[cluster_id → list[CacheEntry]]

    This is the key design: instead of one flat list of all entries,
    we partition entries by their dominant cluster. When a new query
    arrives, we only search the entries in its cluster.

    Thread safety:
        We use a Lock() because FastAPI may handle concurrent requests.
        Without locking, two simultaneous writes could corrupt the cache.
    """

    def __init__(
        self,
        base_threshold: float = 0.82,
        n_clusters: int = 16,
        max_entries_per_cluster: int = 500,
    ):
        """
        Args:
            base_threshold: The base cosine similarity threshold for cache hits.
                           See ADAPTIVE THRESHOLD section above for how this
                           gets adjusted per cluster.

            n_clusters: Must match the number of clusters in your FCM model.

            max_entries_per_cluster: Cap per-cluster cache size.
                                    Prevents unbounded memory growth.
                                    When full, evict the oldest entry (FIFO).
        """
        self.base_threshold = base_threshold
        self.n_clusters = n_clusters
        self.max_entries_per_cluster = max_entries_per_cluster

        # The actual storage: cluster_id → list of CacheEntry
        # defaultdict auto-creates empty lists for new cluster IDs
        self._store: dict[int, list[CacheEntry]] = defaultdict(list)

        # Adaptive thresholds — initialized to base, updated when cluster stats arrive
        self._thresholds: dict[int, float] = {
            c: base_threshold for c in range(n_clusters)
        }

        # Statistics tracking
        self._hit_count: int = 0
        self._miss_count: int = 0

        # Thread lock for concurrent access safety
        self._lock = Lock()

        logger.info(
            f"SemanticCache initialized: base_threshold={base_threshold}, "
            f"n_clusters={n_clusters}"
        )

    # ── Threshold Management ──────────────────────────────────────────────────

    def set_adaptive_thresholds(
        self,
        cluster_variances: dict[int, float],
    ) -> None:
        """
        Set per-cluster thresholds based on intra-cluster variance.

        FORMULA EXPLANATION:
        ─────────────────────
        We compute the z-score of each cluster's variance relative to all clusters:
            z = (variance_c - mean_variance) / std_variance

        Then we adjust the base threshold:
            threshold_c = base + clamp(z * scale, -max_adj, +max_adj)

        Effect:
          High variance cluster → z > 0 → threshold goes UP
            Why? High variance means the cluster is semantically BROAD.
            Broad clusters have more diversity within them.
            We need a higher threshold to avoid matching dissimilar queries
            that happen to land in the same broad cluster.

          Low variance cluster → z < 0 → threshold goes DOWN
            Why? Tight clusters have very similar documents.
            Paraphrased queries will still be very similar.
            Lower threshold = more permissive matching = more cache hits.

        Adjustment is clamped to ±0.08 to avoid extreme thresholds.
        """
        if not cluster_variances:
            return

        variances = np.array(list(cluster_variances.values()))
        mean_var = np.mean(variances)
        std_var = np.std(variances)

        if std_var == 0:
            return  # All clusters have same variance — keep base threshold

        scale = 0.08    # Maximum threshold adjustment in either direction

        for cluster_id, variance in cluster_variances.items():
            z_score = (variance - mean_var) / std_var
            adjustment = np.clip(z_score * scale / 2, -scale, scale)
            new_threshold = np.clip(
                self.base_threshold + adjustment,
                0.70,   # Never go below 0.70 (too many false hits)
                0.95,   # Never go above 0.95 (too restrictive)
            )
            self._thresholds[cluster_id] = float(new_threshold)

        logger.info("Adaptive thresholds set:")
        for c, t in sorted(self._thresholds.items()):
            logger.info(f"  Cluster {c:2d}: threshold = {t:.4f}")

    def get_threshold(self, cluster_id: int) -> float:
        """Return the similarity threshold for a given cluster."""
        return self._thresholds.get(cluster_id, self.base_threshold)

    # ── Core Cache Operations ─────────────────────────────────────────────────

    def lookup(
        self,
        query_vector: np.ndarray,
        dominant_cluster: int,
    ) -> tuple[CacheEntry | None, float]:
        """
        Search the cache for a semantically similar previous query.

        Args:
            query_vector: Normalized embedding of the new query. Shape: (384,)
            dominant_cluster: Which cluster this query belongs to.

        Returns:
            (matched_entry, similarity_score) if cache hit
            (None, 0.0) if cache miss

        ALGORITHM:
        ──────────
        1. Get all cached entries for `dominant_cluster`
        2. For each entry, compute cosine similarity with query_vector
           (cosine similarity = dot product, because vectors are normalized)
        3. Find the entry with maximum similarity
        4. If max_similarity >= threshold → HIT
        5. Otherwise → MISS

        WHY DOT PRODUCT = COSINE SIMILARITY?
        cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
        Since ||a|| = ||b|| = 1 (normalized vectors):
        cosine_similarity(a, b) = a · b = dot_product(a, b)
        """
        with self._lock:
            cluster_entries = self._store.get(dominant_cluster, [])

            if not cluster_entries:
                self._miss_count += 1
                return None, 0.0

            threshold = self.get_threshold(dominant_cluster)

            # Compute cosine similarity to all cached entries in this cluster
            # Stack all cached vectors into a matrix for efficient batch computation
            cached_vectors = np.vstack([e.query_vector for e in cluster_entries])

            # Batch dot product: shape (N_cached,)
            # Each value is the cosine similarity between query and one cached entry
            similarities = cached_vectors @ query_vector

            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])

            if best_similarity >= threshold:
                # CACHE HIT
                matched_entry = cluster_entries[best_idx]
                matched_entry.record_hit()
                self._hit_count += 1
                logger.debug(
                    f"Cache HIT: similarity={best_similarity:.4f} "
                    f"(threshold={threshold:.4f}, cluster={dominant_cluster})"
                )
                return matched_entry, best_similarity
            else:
                # CACHE MISS
                self._miss_count += 1
                logger.debug(
                    f"Cache MISS: best_similarity={best_similarity:.4f} "
                    f"(threshold={threshold:.4f}, cluster={dominant_cluster})"
                )
                return None, best_similarity

    def store(
        self,
        query_text: str,
        query_vector: np.ndarray,
        result: str,
        dominant_cluster: int,
        cluster_memberships: np.ndarray,
    ) -> CacheEntry:
        """
        Store a new query-result pair in the cache.

        On capacity overflow, we evict the OLDEST entry (FIFO eviction).
        Alternative policies (LRU, LFU) could be better but add complexity.
        FIFO is correct here because older cached queries may be stale
        and newer queries are more likely to be repeated.
        """
        entry = CacheEntry(
            query_text=query_text,
            query_vector=query_vector.copy(),  # Copy to prevent mutation
            result=result,
            dominant_cluster=dominant_cluster,
            cluster_memberships=cluster_memberships.copy(),
        )

        with self._lock:
            cluster_entries = self._store[dominant_cluster]

            # Evict oldest if at capacity
            if len(cluster_entries) >= self.max_entries_per_cluster:
                evicted = cluster_entries.pop(0)  # Remove oldest (index 0)
                logger.debug(f"Evicted oldest entry from cluster {dominant_cluster}")

            cluster_entries.append(entry)

        logger.debug(
            f"Stored entry: cluster={dominant_cluster}, "
            f"total_in_cluster={len(self._store[dominant_cluster])}"
        )
        return entry

    # ── Cache Management ──────────────────────────────────────────────────────

    def flush(self) -> None:
        """
        Clear the entire cache and reset all statistics.
        Called by DELETE /cache endpoint.
        """
        with self._lock:
            self._store.clear()
            self._hit_count = 0
            self._miss_count = 0
        logger.info("Cache flushed. All entries and statistics reset.")

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return cache statistics for the GET /cache/stats endpoint.
        Extended stats beyond the required spec include cluster_distribution.
        """
        with self._lock:
            total_entries = sum(len(v) for v in self._store.values())
            total_queries = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_queries if total_queries > 0 else 0.0

            # Per-cluster breakdown of cache entries
            cluster_distribution = {
                str(k): len(v)
                for k, v in self._store.items()
                if len(v) > 0
            }

            # Threshold info (useful for debugging)
            threshold_info = {
                str(k): round(v, 4)
                for k, v in self._thresholds.items()
            }

            return {
                "total_entries": total_entries,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(hit_rate, 4),
                "cluster_distribution": cluster_distribution,
                "adaptive_thresholds": threshold_info,
            }

    def get_all_entries(self) -> list[CacheEntry]:
        """Return all entries across all clusters (for analysis/debugging)."""
        with self._lock:
            all_entries = []
            for entries in self._store.values():
                all_entries.extend(entries)
        return all_entries