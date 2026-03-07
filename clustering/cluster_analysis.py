"""
clustering/cluster_analysis.py

This module answers three questions:
  1. How many clusters K should we use? (justified with metrics)
  2. What do the clusters mean? (semantic labeling)
  3. Which documents are boundary cases? (entropy analysis)

This is the "analysis" layer — nothing is trained here, only interpreted.
"""

import logging
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ── K Selection ───────────────────────────────────────────────────────────────

def sweep_k_values(
    embeddings: np.ndarray,
    k_values: list[int],
    fuzziness: float = 2.0,
) -> list[dict]:
    """
    Fit FCM for multiple values of K and compute evaluation metrics.
    Use the results to pick the best K.

    METRICS WE USE:
    ─────────────────
    1. Silhouette Score (-1 to 1, higher = better separation)
       Measures: how similar is a doc to its own cluster vs other clusters?
       Best K: the one with the highest silhouette score.

    2. Mean Entropy (lower = better defined clusters)
       Low entropy means most documents clearly belong to one cluster.
       High entropy means everything is blurry (too few clusters).

    3. Fuzzy Partition Coefficient (0 to 1, higher = less fuzzy)
       FPC = 1 means hard clusters, FPC = 1/K means total overlap.

    THE TRADEOFF:
    More clusters → lower entropy (docs are more specific)
                  → lower silhouette (harder to separate so many clusters)
    We want the "elbow" — the K where we gain diminishing returns.
    """
    from clustering.fuzzy_cluster import FuzzyClusterer

    results = []
    # Use a subset for speed during sweep (silhouette is O(N^2))
    sample_size = min(3000, len(embeddings))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(embeddings), sample_size, replace=False)
    sample = embeddings[sample_idx]

    for k in k_values:
        logger.info(f"Testing K={k}...")
        clusterer = FuzzyClusterer(n_clusters=k, fuzziness=fuzziness)
        clusterer.fit(sample)

        # Hard labels for silhouette (silhouette needs hard assignments)
        hard_labels = np.argmax(clusterer.memberships_, axis=1)

        # Silhouette score — skip if all docs in one cluster
        unique_labels = np.unique(hard_labels)
        if len(unique_labels) < 2:
            sil = -1.0
        else:
            # Use subset for speed
            sil_sample = min(500, len(sample))
            sil_idx = rng.choice(len(sample), sil_sample, replace=False)
            sil = silhouette_score(
                sample[sil_idx],
                hard_labels[sil_idx],
                metric='cosine',
            )

        mean_entropy = float(np.mean(clusterer.entropy_per_doc_))
        max_entropy = np.log2(k)  # Theoretical maximum entropy for K clusters

        results.append({
            'k': k,
            'silhouette': round(sil, 4),
            'mean_entropy': round(mean_entropy, 4),
            'normalized_entropy': round(mean_entropy / max_entropy, 4),
            # normalized entropy: 0=perfectly separated, 1=completely fuzzy
        })

        logger.info(
            f"  K={k}: silhouette={sil:.4f}, "
            f"mean_entropy={mean_entropy:.4f}, "
            f"norm_entropy={mean_entropy/max_entropy:.4f}"
        )

    return results


def pick_best_k(sweep_results: list[dict]) -> int:
    """
    Given sweep results, pick the K with the best tradeoff.

    Strategy: maximize (silhouette) while keeping normalized_entropy < 0.6
    If multiple K values have good silhouette, prefer the one with lower entropy.
    This balances cluster quality vs semantic specificity.
    """
    # Filter to reasonable entropy range
    candidates = [r for r in sweep_results if r['normalized_entropy'] < 0.65]
    if not candidates:
        candidates = sweep_results  # Fallback: use all

    # Pick highest silhouette among candidates
    best = max(candidates, key=lambda x: x['silhouette'])
    logger.info(
        f"Selected K={best['k']} "
        f"(silhouette={best['silhouette']}, entropy={best['mean_entropy']})"
    )
    return best['k']


# ── Cluster Labeling ──────────────────────────────────────────────────────────

# Manual semantic labels based on what we know about 20 newsgroups.
# These are assigned by inspecting the top documents per cluster after fitting.
# The keys are cluster IDs; you update these after running the analysis.
CLUSTER_LABELS: dict[int, str] = {
    # These are placeholder labels — update after inspecting cluster contents
    0:  "unknown",
    1:  "unknown",
    2:  "unknown",
    3:  "unknown",
    4:  "unknown",
    5:  "unknown",
    6:  "unknown",
    7:  "unknown",
    8:  "unknown",
    9:  "unknown",
    10: "unknown",
    11: "unknown",
    12: "unknown",
    13: "unknown",
    14: "unknown",
    15: "unknown",
}


def label_clusters_automatically(
    clusterer,
    documents: list,
    top_docs_per_cluster: int = 30,
) -> dict[int, str]:
    """
    Automatically suggest cluster labels by looking at the most common
    original newsgroup categories among each cluster's top documents.

    This is how we VALIDATE our unsupervised clusters against known labels.
    If cluster 7 is mostly 'talk.politics.guns' and 'talk.politics.misc',
    we label it 'firearms_politics'.

    Args:
        clusterer: Fitted FuzzyClusterer
        documents: List of Document objects (must have original_category)
        top_docs_per_cluster: How many strongest-membership docs to examine per cluster

    Returns:
        dict mapping cluster_id → suggested_label
    """
    suggested_labels = {}

    for c in range(clusterer.n_clusters):
        # Get membership scores for cluster c
        cluster_memberships = clusterer.memberships_[:, c]

        # Find indices of top documents for this cluster
        top_indices = np.argsort(cluster_memberships)[::-1][:top_docs_per_cluster]

        # Count original categories among these top docs
        category_counts = Counter(
            documents[i].original_category for i in top_indices
        )

        # The top 2 categories define the cluster's semantic meaning
        top_2 = category_counts.most_common(2)
        if top_2:
            # Clean up category names: 'talk.politics.guns' → 'politics_guns'
            label_parts = []
            for cat, _ in top_2:
                # Take the last two parts of the dotted name
                parts = cat.split('.')
                label_parts.append('_'.join(parts[-2:]) if len(parts) >= 2 else cat)
            suggested_labels[c] = ' + '.join(label_parts)
        else:
            suggested_labels[c] = f'cluster_{c}'

    return suggested_labels


def analyze_cluster_composition(
    clusterer,
    documents: list,
) -> dict[int, dict]:
    """
    For each cluster, show:
      - The distribution of original newsgroup categories (validation)
      - Average membership strength
      - Whether the cluster is "pure" (one category) or "mixed" (multiple)

    This is the key analysis that proves our clusters are semantically meaningful.
    A good cluster should map closely to 1-2 original categories.
    A bad cluster would have 10+ categories equally distributed.
    """
    cluster_analysis = defaultdict(lambda: {
        'category_distribution': Counter(),
        'total_membership': 0.0,
        'doc_count': 0,
    })

    for doc_idx, doc in enumerate(documents):
        dominant_c = clusterer.get_dominant_cluster(doc_idx)
        membership_strength = float(clusterer.memberships_[doc_idx, dominant_c])

        cluster_analysis[dominant_c]['category_distribution'][doc.original_category] += 1
        cluster_analysis[dominant_c]['total_membership'] += membership_strength
        cluster_analysis[dominant_c]['doc_count'] += 1

    # Compute purity for each cluster
    results = {}
    for c, data in cluster_analysis.items():
        total = data['doc_count']
        if total == 0:
            continue

        top_category, top_count = data['category_distribution'].most_common(1)[0]
        purity = top_count / total  # 1.0 = all docs from same category (perfect)

        results[c] = {
            'dominant_count': total,
            'top_category': top_category,
            'purity': round(purity, 3),
            'avg_membership': round(data['total_membership'] / total, 3),
            'category_distribution': dict(data['category_distribution'].most_common(5)),
        }

    return results


# ── Boundary Document Analysis ────────────────────────────────────────────────

def find_boundary_documents(
    clusterer,
    documents: list,
    top_n: int = 20,
) -> list[dict]:
    """
    Find and describe the most semantically ambiguous documents.

    WHAT MAKES A BOUNDARY DOCUMENT INTERESTING?
    These are documents that genuinely bridge multiple topics.
    They're not classification errors — they're evidence that real-world
    text doesn't respect clean category boundaries.

    Example output:
      Doc #4821: "Re: assault weapons ban in California"
        - cluster_2 (politics):  0.38
        - cluster_7 (firearms):  0.35
        - cluster_11 (law):      0.27
        - entropy: 1.58 (high — very uncertain)
        - original_category: talk.politics.guns (makes sense!)
    """
    doc_ids = [d.doc_id for d in documents]
    boundary_raw = clusterer.get_boundary_documents(doc_ids, top_n=top_n)

    enriched = []
    for b in boundary_raw:
        pos = b['position']
        doc = documents[pos]

        enriched.append({
            'doc_id': doc.doc_id,
            'original_category': doc.original_category,
            'entropy': b['entropy'],
            'top_3_clusters': b['top_3_clusters'],
            'text_preview': doc.text[:300],
        })

    return enriched


def compute_cluster_variance(clusterer) -> dict[int, float]:
    """
    Compute intra-cluster variance for each cluster.
    Used by the semantic cache to set adaptive similarity thresholds.

    HIGH variance cluster:
      Documents are spread out in vector space → cluster is semantically broad
      → Use LOWER threshold (accept more cache hits, slightly less precise)

    LOW variance cluster:
      Documents are tightly packed → cluster is semantically specific
      → Use HIGHER threshold (only match very similar queries)
    """
    variances = {}
    for c in range(clusterer.n_clusters):
        # Get all documents with meaningful membership in this cluster
        # (membership > 0.1 means the cluster is relevant to this doc)
        mask = clusterer.memberships_[:, c] > 0.1

        if mask.sum() < 2:
            variances[c] = 0.1  # Default for near-empty clusters
            continue

        # Weighted variance: use membership as weights
        weights = clusterer.memberships_[mask, c]

        # We need the actual embeddings for variance computation
        # This is called from the cache module where embeddings are available
        variances[c] = float(np.var(weights))  # Placeholder — real impl in cache

    return variances