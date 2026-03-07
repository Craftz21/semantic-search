"""
clustering/fuzzy_cluster.py  (v6 — Spherical FCM with cosine distance)

WHY ALL PREVIOUS FCM ATTEMPTS FAILED:
───────────────────────────────────────
Every version used Euclidean distance between vectors.
Sentence embeddings (even after PCA) are cosine-similarity spaces.

The problem:
  - After PCA, vectors have different magnitudes but similar directions
  - Euclidean distance conflates magnitude with direction
  - Cluster centers end up at similar Euclidean distances from most points
  - FCM update: u_ki = 1/Σ(d_ki/d_ji)^exp → all ratios ≈ 1 → u_ki = 1/K

THE FIX: Spherical FCM
────────────────────────
Use COSINE DISTANCE instead of Euclidean distance.

cosine_distance(a, b) = 1 - cosine_similarity(a, b)
                      = 1 - (a·b)/(||a||·||b||)

For text embeddings, cosine distances between different-topic clusters
are meaningfully different (0.3–0.8), giving FCM strong gradient.

Spherical FCM also normalizes cluster centers after each update:
  c_k = normalize(Σ u_ki^m * x_i)
This keeps centers on the unit hypersphere, consistent with the metric.

This is a well-established adaptation of FCM for text and NLP data.
References: Dhillon & Modha (2001) "Concept Decompositions for Large
Sparse Text Data Using Clustering"
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)


class FuzzyClusterer:
    """
    Spherical Fuzzy C-Means: FCM with cosine distance.
    Produces genuine soft membership distributions over clusters.
    """

    def __init__(
        self,
        n_clusters: int = 16,
        fuzziness: float = 2.0,
        max_iter: int = 150,
        pca_components: int = 50,
    ):
        """
        fuzziness (m) = 2.0:
            Standard FCM parameter. With cosine distance (which gives
            discriminative values in [0,2]), m=2.0 produces good soft
            assignments without degeneracy.

        pca_components = 50:
            Reduce dims before FCM. After PCA we L2-normalize so vectors
            sit on a hypersphere — consistent with cosine distance metric.
        """
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.max_iter = max_iter
        self.pca_components = pca_components
        self.error_threshold = 1e-4

        self.pca_: PCA | None = None
        self.cluster_centers_: np.ndarray | None = None   # (K, pca_components), L2-normalized
        self.memberships_: np.ndarray | None = None        # (N, K)
        self.entropy_per_doc_: np.ndarray | None = None

        logger.info(
            f"FuzzyClusterer v6 (Spherical FCM): "
            f"K={n_clusters}, m={fuzziness}, PCA→{pca_components}dims, cosine distance"
        )

    def fit(self, embeddings: np.ndarray) -> "FuzzyClusterer":
        N, original_dim = embeddings.shape
        logger.info(f"Input: {N} docs × {original_dim} dims")

        # ── Step 1: PCA ───────────────────────────────────────────────────────
        logger.info(f"PCA: {original_dim} → {self.pca_components} dims...")
        self.pca_ = PCA(
            n_components=self.pca_components,
            random_state=42,
            svd_solver='randomized',
        )
        reduced = self.pca_.fit_transform(embeddings)
        variance_retained = self.pca_.explained_variance_ratio_.sum()
        logger.info(f"Variance retained: {variance_retained:.1%}")

        # ── Step 2: L2-normalize AFTER PCA ────────────────────────────────────
        # Critical: normalize so cosine_distance = 1 - dot_product
        # This puts all points on the unit hypersphere in PCA space.
        # Spherical FCM requires this — centers are also kept normalized.
        data = self._normalize_rows(reduced).astype(np.float64)
        logger.info("L2-normalized PCA output (spherical representation)")

        # ── Step 3: KMeans warm-start (spherical KMeans) ──────────────────────
        logger.info(f"KMeans warm-start (K={self.n_clusters})...")
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        kmeans.fit(data)
        sizes = np.bincount(kmeans.labels_).tolist()
        logger.info(f"KMeans sizes: {sizes}")

        # Build init membership matrix (K, N) from KMeans labels
        init_U = np.full(
            (self.n_clusters, N),
            fill_value=0.09 / max(self.n_clusters - 1, 1),
            dtype=np.float64,
        )
        for i, label in enumerate(kmeans.labels_):
            init_U[label, i] = 0.91

        # ── Step 4: Spherical FCM ─────────────────────────────────────────────
        logger.info("Running Spherical FCM (cosine distance)...")
        centers, U = self._spherical_fcm(data, init_U)

        self.cluster_centers_ = centers.astype(np.float32)   # (K, pca_components)
        self.memberships_ = U.T.astype(np.float32)            # (N, K)

        self._compute_entropy()
        self._sanity_check()
        return self

    def _spherical_fcm(
        self,
        data: np.ndarray,    # (N, features) — L2-normalized rows
        init_U: np.ndarray,  # (K, N)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Spherical FCM algorithm using cosine distance.

        COSINE DISTANCE: d(x, c) = 1 - x·c  (for normalized vectors)
          Range: [0, 2] where 0=identical, 2=opposite
          Much more discriminative than Euclidean for text embeddings.

        CENTER UPDATE (spherical):
          c_k = normalize(Σ_i u_ki^m * x_i)
          Normalization keeps centers on the unit sphere.

        MEMBERSHIP UPDATE (standard FCM formula with cosine distance):
          u_ki = 1 / Σ_j (d_ki / d_ji)^(2/(m-1))
        """
        U = init_U.copy()
        N, features = data.shape
        K = self.n_clusters
        m = self.fuzziness
        exp = 2.0 / (m - 1.0)

        for iteration in range(self.max_iter):
            U_old = U.copy()

            # ── Update centers (spherical: normalize after weighted sum) ──────
            # Um[k,i] = u_ki^m
            Um = U ** m                                   # (K, N)
            weight_sums = Um.sum(axis=1, keepdims=True)   # (K, 1)
            centers = Um @ data / weight_sums             # (K, features)
            centers = self._normalize_rows(centers)       # Keep on unit sphere

            # ── Cosine distances: d[k,i] = 1 - centers[k] · data[i] ─────────
            # dot_products[k,i] = centers[k] · data[i]  (both normalized)
            dot_products = centers @ data.T               # (K, N)
            dists = 1.0 - dot_products                    # cosine distance (K, N)

            # Clamp: cosine distance can be negative due to floating point
            dists = np.fmax(dists, 1e-10)

            # ── Update memberships ────────────────────────────────────────────
            # u_ki = 1 / Σ_j (d_ki/d_ji)^exp
            # dist_ratios[k,j,i] = d_ki / d_ji
            # = dists[k,:] / dists[j,:]  broadcast over all j
            dist_ratios = (dists[:, np.newaxis, :] / dists[np.newaxis, :, :]) ** exp
            # Sum over j dimension → (K, N)
            U = 1.0 / dist_ratios.sum(axis=1)

            # ── Convergence ───────────────────────────────────────────────────
            delta = float(np.max(np.abs(U - U_old)))
            if iteration % 10 == 0:
                max_mem = float(U.max())
                logger.info(
                    f"  iter {iteration:3d}: delta={delta:.6f}, "
                    f"max_membership={max_mem:.4f}"
                )
            if delta < self.error_threshold:
                logger.info(f"  ✅ Converged at iteration {iteration+1}")
                break
        else:
            logger.warning(f"FCM did not fully converge in {self.max_iter} iterations")

        return centers, U

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row to unit length."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def _compute_entropy(self) -> None:
        self.entropy_per_doc_ = np.array([
            scipy_entropy(mem + 1e-10, base=2)
            for mem in self.memberships_
        ])
        mean_e = float(self.entropy_per_doc_.mean())
        max_e = np.log2(self.n_clusters)
        pct = mean_e / max_e

        if pct < 0.60:
            status = "✅ GOOD"
        elif pct < 0.80:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ DEGENERATE"

        logger.info(
            f"Mean entropy: {mean_e:.4f} / {max_e:.4f} "
            f"({pct:.1%} of max)  {status}"
        )

    def _sanity_check(self) -> None:
        counts = np.bincount(
            np.argmax(self.memberships_, axis=1),
            minlength=self.n_clusters,
        )
        logger.info("Cluster dominant-doc counts:")
        for c, cnt in enumerate(counts):
            bar = "█" * max(0, cnt // 150)
            logger.info(f"  Cluster {c:2d}: {cnt:5d}  {bar}")

        sample = self.memberships_[0]
        top3 = np.argsort(sample)[::-1][:3]
        logger.info("Sample doc #0 top-3 memberships:")
        for c in top3:
            bar = "█" * int(sample[c] * 40)
            logger.info(f"  cluster_{c}: {sample[c]:.4f}  {bar}")

    def predict_single(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Get soft cluster memberships for a new query.
        Apply same PCA + normalization, then one FCM membership update.
        """
        if self.pca_ is None or self.cluster_centers_ is None:
            raise RuntimeError("Must call fit() first")

        reduced = self.pca_.transform(query_vector.reshape(1, -1))
        normalized = self._normalize_rows(reduced.astype(np.float64))  # (1, features)

        centers = self.cluster_centers_.astype(np.float64)  # (K, features)
        m = self.fuzziness
        exp = 2.0 / (m - 1.0)

        # Cosine distances from query to each center
        dot_products = centers @ normalized.T   # (K, 1)
        dists = 1.0 - dot_products[:, 0]        # (K,)
        dists = np.fmax(dists, 1e-10)

        # Membership update
        ratios = (dists[:, np.newaxis] / dists[np.newaxis, :]) ** exp  # (K, K)
        memberships = 1.0 / ratios.sum(axis=1)  # (K,)

        return memberships.astype(np.float32)

    def get_dominant_cluster(self, doc_idx: int) -> int:
        return int(np.argmax(self.memberships_[doc_idx]))

    def get_top_clusters(self, doc_idx: int, top_n: int = 3) -> list[tuple[int, float]]:
        mem = self.memberships_[doc_idx]
        top_indices = np.argsort(mem)[::-1][:top_n]
        return [(int(i), float(mem[i])) for i in top_indices]

    def get_cluster_summary(self) -> list[dict]:
        summaries = []
        for c in range(self.n_clusters):
            dominant_count = int(np.sum(np.argmax(self.memberships_, axis=1) == c))
            mc = self.memberships_[:, c]
            summaries.append({
                'cluster_id': c,
                'dominant_doc_count': dominant_count,
                'mean_membership': float(mc.mean()),
                'max_membership': float(mc.max()),
                'std_membership': float(mc.std()),
            })
        return sorted(summaries, key=lambda x: x['dominant_doc_count'], reverse=True)

    def get_boundary_documents(self, doc_ids: list[int], top_n: int = 20) -> list[dict]:
        top_positions = np.argsort(self.entropy_per_doc_)[::-1][:top_n]
        results = []
        for pos in top_positions:
            results.append({
                'position': int(pos),
                'doc_id': doc_ids[pos] if doc_ids else int(pos),
                'entropy': float(self.entropy_per_doc_[pos]),
                'top_3_clusters': self.get_top_clusters(int(pos), 3),
                'dominant_cluster': self.get_dominant_cluster(int(pos)),
            })
        return results

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FuzzyClusterer":
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Loaded FuzzyClusterer (K={obj.n_clusters})")
        return obj