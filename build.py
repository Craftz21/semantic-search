"""
build.py

Run this ONCE before starting the API server.
This script:
  1. Fetches and cleans the dataset
  2. Embeds all documents
  3. Fits the fuzzy clustering model
  4. Saves everything to disk

After this runs, the API server loads from disk on startup.
That way the server starts in seconds, not minutes.

Usage:
  python build.py
  python build.py --n-clusters 20    # Override K
  python build.py --skip-embed       # Reuse saved embeddings (re-run clustering only)
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

EMBEDDINGS_PATH   = MODELS_DIR / "embeddings.npy"
METADATA_PATH     = MODELS_DIR / "metadata.pkl"
FAISS_INDEX_PATH  = MODELS_DIR / "faiss.index"
CLUSTER_MODEL_PATH = MODELS_DIR / "cluster_model.pkl"
DOCUMENTS_PATH    = MODELS_DIR / "documents.pkl"
CLUSTER_LABELS_PATH = MODELS_DIR / "cluster_labels.pkl"


def main(n_clusters: int, skip_embed: bool):
    logger.info("=" * 60)
    logger.info("SEMANTIC SEARCH SYSTEM — BUILD SCRIPT")
    logger.info("=" * 60)

    # ── Step 1: Fetch and clean dataset ──────────────────────────────────────
    logger.info("\n[STEP 1] Fetching and cleaning dataset...")
    from data.fetch_and_clean import fetch_and_clean_dataset
    documents, category_names = fetch_and_clean_dataset(subset="all")

    # Save documents for later use by API
    with open(DOCUMENTS_PATH, 'wb') as f:
        pickle.dump({'documents': documents, 'category_names': category_names}, f)
    logger.info(f"Saved {len(documents)} documents to {DOCUMENTS_PATH}")

    texts = [doc.text for doc in documents]

    # ── Step 2: Embed documents ───────────────────────────────────────────────
    if skip_embed and EMBEDDINGS_PATH.exists():
        logger.info("\n[STEP 2] Loading existing embeddings (--skip-embed flag set)...")
        embeddings = np.load(str(EMBEDDINGS_PATH))
        logger.info(f"Loaded embeddings: {embeddings.shape}")
    else:
        logger.info("\n[STEP 2] Embedding documents...")
        logger.info("  This takes 5-15 minutes on CPU. Grab a coffee.")
        from embeddings.embedder import TextEmbedder
        embedder = TextEmbedder(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        embeddings = embedder.embed_batch(texts, batch_size=64, show_progress=True)
        embedder.save_embeddings(embeddings, str(EMBEDDINGS_PATH))
        logger.info(f"Embeddings saved: {embeddings.shape}")

    # ── Step 3: Build FAISS index ─────────────────────────────────────────────
    logger.info("\n[STEP 3] Building FAISS vector store...")
    from vector_store.faiss_store import FAISSVectorStore
    store = FAISSVectorStore(dimension=embeddings.shape[1])

    # Prepare metadata (we'll update dominant_cluster after clustering)
    metadata_list = [
        {
            'doc_id': doc.doc_id,
            'text': doc.text[:500],   # Store first 500 chars for retrieval preview
            'category': doc.original_category,
            'category_id': doc.original_category_id,
            'dominant_cluster': -1,   # Will be filled after clustering
            'cluster_memberships': None,
        }
        for doc in documents
    ]

    store.add_documents(embeddings, metadata_list)
    logger.info(f"FAISS index built: {store.size} vectors")

    # ── Step 4: Fuzzy Clustering ──────────────────────────────────────────────
    logger.info(f"\n[STEP 4] Running Spherical FCM with K={n_clusters}...")
    logger.info("  Spherical FCM takes 5-15 minutes on CPU.")
    from clustering.fuzzy_cluster import FuzzyClusterer

    clusterer = FuzzyClusterer(
        n_clusters=n_clusters,
        fuzziness=2.0,
        max_iter=150,
        pca_components=50,
    )
    clusterer.fit(embeddings)

    # ── Step 5: Update metadata with cluster assignments ──────────────────────
    logger.info("\n[STEP 5] Updating metadata with cluster assignments...")
    for i, meta in enumerate(store.metadata):
        dominant = clusterer.get_dominant_cluster(i)
        meta['dominant_cluster'] = dominant
        meta['cluster_memberships'] = clusterer.memberships_[i].tolist()

    # ── Step 6: Auto-label clusters ───────────────────────────────────────────
    logger.info("\n[STEP 6] Auto-labeling clusters...")
    from clustering.cluster_analysis import label_clusters_automatically
    cluster_labels = label_clusters_automatically(clusterer, documents)

    logger.info("Suggested cluster labels:")
    for c, label in sorted(cluster_labels.items()):
        summary = [
            s for s in clusterer.get_cluster_summary()
            if s['cluster_id'] == c
        ]
        count = summary[0]['dominant_doc_count'] if summary else 0
        logger.info(f"  Cluster {c:2d}: {label:<40} ({count} docs as dominant)")

    # ── Step 7: Compute cluster variances for adaptive thresholds ─────────────
    logger.info("\n[STEP 7] Computing cluster variances for adaptive thresholds...")
    cluster_variances = {}
    for c in range(n_clusters):
        mask = clusterer.memberships_[:, c] > 0.1
        if mask.sum() >= 2:
            cluster_embeddings = embeddings[mask]
            cluster_weights = clusterer.memberships_[mask, c]
            # Weighted variance: documents with higher membership count more
            weighted_mean = np.average(cluster_embeddings, axis=0, weights=cluster_weights)
            diffs = cluster_embeddings - weighted_mean
            weighted_var = np.average(np.sum(diffs**2, axis=1), weights=cluster_weights)
            cluster_variances[c] = float(weighted_var)
        else:
            cluster_variances[c] = 0.01

    logger.info("Cluster variances (affects cache thresholds):")
    for c, var in sorted(cluster_variances.items()):
        logger.info(f"  Cluster {c:2d}: variance={var:.6f}")

    # ── Step 8: Save everything ───────────────────────────────────────────────
    logger.info("\n[STEP 8] Saving all models to disk...")
    store.save(str(FAISS_INDEX_PATH), str(METADATA_PATH))
    clusterer.save(str(CLUSTER_MODEL_PATH))

    with open(CLUSTER_LABELS_PATH, 'wb') as f:
        pickle.dump({
            'labels': cluster_labels,
            'variances': cluster_variances,
            'n_clusters': n_clusters,
        }, f)

    logger.info("\n" + "=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info(f"  Documents indexed: {len(documents)}")
    logger.info(f"  Clusters:          {n_clusters}")
    logger.info(f"  Models saved to:   {MODELS_DIR}/")
    logger.info("\nNext step:")
    logger.info("  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build semantic search system")
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--skip-embed", action="store_true",
                        help="Skip embedding step if embeddings already saved")
    args = parser.parse_args()
    main(args.n_clusters, args.skip_embed)