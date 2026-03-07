"""
api/main.py

FastAPI service for the Semantic Search system.

Endpoints:
POST   /query        → semantic search
GET    /cache/stats  → cache analytics
DELETE /cache        → flush cache
GET    /health       → service health
GET    /             → root info

Startup Flow
------------
1. Server starts
2. Models load once during lifespan startup
3. Requests reuse models in memory

State Management
----------------
All models and cache objects live at module level and are
loaded only once. This keeps request latency low.
"""

import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# ───────────────── Logging ─────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# ───────────────── Paths ─────────────────

MODELS_DIR = Path("models")

FAISS_INDEX_PATH = MODELS_DIR / "faiss.index"
METADATA_PATH = MODELS_DIR / "metadata.pkl"
CLUSTER_MODEL_PATH = MODELS_DIR / "cluster_model.pkl"
DOCUMENTS_PATH = MODELS_DIR / "documents.pkl"
CLUSTER_LABELS_PATH = MODELS_DIR / "cluster_labels.pkl"

# ───────────────── Global State ─────────────────

embedder = None
vector_store = None
clusterer = None
cache = None
cluster_labels = {}
documents = []

# ───────────────── Startup / Shutdown ─────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models during server startup.
    """

    global embedder, vector_store, clusterer, cache, cluster_labels, documents

    logger.info("Starting up: loading models...")
    start = time.time()

    if not CLUSTER_MODEL_PATH.exists():
        raise RuntimeError(
            "Models not found. Run `python build.py` before starting the API."
        )

    # ── Load embedding model ─────────────────

    from embeddings.embedder import TextEmbedder

    embedder = TextEmbedder(
        model_name=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    # ── Load FAISS vector store ─────────────────

    from vector_store.faiss_store import FAISSVectorStore

    vector_store = FAISSVectorStore.load(
        str(FAISS_INDEX_PATH),
        str(METADATA_PATH),
    )

    # ── Load cluster model ─────────────────

    from clustering.fuzzy_cluster import FuzzyClusterer

    clusterer = FuzzyClusterer.load(str(CLUSTER_MODEL_PATH))

    # ── Load cluster labels ─────────────────

    with open(CLUSTER_LABELS_PATH, "rb") as f:
        label_data = pickle.load(f)

    cluster_labels = label_data["labels"]
    cluster_variances = label_data["variances"]
    n_clusters = label_data["n_clusters"]

    # ── Load documents ─────────────────

    with open(DOCUMENTS_PATH, "rb") as f:
        doc_data = pickle.load(f)

    documents = doc_data["documents"]

    # ── Initialize semantic cache ─────────────────

    from cache.semantic_cache import SemanticCache

    cache = SemanticCache(
        base_threshold=float(
            os.getenv("BASE_SIMILARITY_THRESHOLD", "0.82")
        ),
        n_clusters=n_clusters,
    )

    cache.set_adaptive_thresholds(cluster_variances)

    elapsed = time.time() - start

    logger.info(f"Startup complete in {elapsed:.1f}s")
    logger.info(f"Vector store size: {vector_store.size}")
    logger.info(f"Clusters: {n_clusters}")
    logger.info(f"Cache threshold: {cache.base_threshold}")

    yield

    logger.info("Shutting down server...")


# ───────────────── FastAPI App ─────────────────

app = FastAPI(
    title="Semantic Search API",
    description="Cluster-aware semantic search with fuzzy clustering and adaptive caching",
    version="1.0.0",
    lifespan=lifespan,
)

# ───────────────── Request Models ─────────────────


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str
    dominant_cluster: int
    cluster_label: str
    cluster_entropy: float
    membership_vector: dict[str, float]
    cache_threshold_used: float
    processing_time_ms: float


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    cluster_distribution: dict[str, int]
    adaptive_thresholds: dict[str, float]


# ───────────────── Result Builder ─────────────────


def synthesize_result(query, top_docs, dominant_cluster, cluster_label):

    if not top_docs:
        return f"No relevant documents found for query '{query}'"

    cluster_name = cluster_label or f"cluster_{dominant_cluster}"

    lines = [
        f"Query routed to semantic cluster: {cluster_name}",
        f"Top {len(top_docs)} relevant documents:",
        "",
    ]

    for i, doc in enumerate(top_docs[:5], 1):

        similarity = doc.get("similarity", 0)
        category = doc.get("category", "unknown")
        text_preview = doc.get("text", "")[:200].replace("\n", " ")

        lines.append(
            f"[{i}] similarity={similarity:.3f} | category={category}"
        )

        lines.append(f"    {text_preview}...")
        lines.append("")

    return "\n".join(lines)


# ───────────────── Query Endpoint ─────────────────


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):

    start = time.time()

    query_text = request.query.strip()

    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # ── Step 1: Embed query ─────────────────

    try:
        query_vector = embedder.embed_single(query_text)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding failed")

    # ── Step 2: Predict cluster ─────────────────

    memberships = clusterer.predict_single(query_vector)

    dominant_cluster = int(np.argmax(memberships))

    cluster_entropy = float(
        -np.sum(memberships * np.log2(memberships + 1e-10))
    )

    top3 = np.argsort(memberships)[::-1][:3]

    membership_vector = {
        f"cluster_{i}": round(float(memberships[i]), 4)
        for i in top3
    }

    cluster_label = cluster_labels.get(
        dominant_cluster, f"cluster_{dominant_cluster}"
    )

    threshold_used = cache.get_threshold(dominant_cluster)

    # ── Step 3: Cache lookup ─────────────────

    cached_entry, similarity = cache.lookup(
        query_vector, dominant_cluster
    )

    if cached_entry is not None:

        elapsed = (time.time() - start) * 1000

        logger.info(
            f"CACHE HIT | query='{query_text[:50]}' "
            f"| similarity={similarity:.4f} "
            f"| cluster={dominant_cluster} "
            f"| {elapsed:.1f}ms"
        )

        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=cached_entry.query_text,
            similarity_score=round(similarity, 4),
            result=cached_entry.result,
            dominant_cluster=dominant_cluster,
            cluster_label=cluster_label,
            cluster_entropy=round(cluster_entropy, 4),
            membership_vector=membership_vector,
            cache_threshold_used=round(threshold_used, 4),
            processing_time_ms=round(elapsed, 2),
        )

    # ── Cache miss → search FAISS ─────────────────

    top_docs = vector_store.search(
        query_vector=query_vector,
        top_k=10,
        cluster_filter=dominant_cluster,
    )

    if len(top_docs) < 3:

        top_docs = vector_store.search(
            query_vector=query_vector,
            top_k=10,
            cluster_filter=None,
        )

    result = synthesize_result(
        query_text,
        top_docs,
        dominant_cluster,
        cluster_label,
    )

    # ── Store in cache ─────────────────

    cache.store(
        query_text=query_text,
        query_vector=query_vector,
        result=result,
        dominant_cluster=dominant_cluster,
        cluster_memberships=memberships,
    )

    elapsed = (time.time() - start) * 1000

    logger.info(
        f"CACHE MISS | query='{query_text[:50]}' "
        f"| cluster={dominant_cluster} "
        f"| docs={len(top_docs)} "
        f"| {elapsed:.1f}ms"
    )

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=dominant_cluster,
        cluster_label=cluster_label,
        cluster_entropy=round(cluster_entropy, 4),
        membership_vector=membership_vector,
        cache_threshold_used=round(threshold_used, 4),
        processing_time_ms=round(elapsed, 2),
    )


# ───────────────── Cache Stats ─────────────────


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():

    stats = cache.get_stats()

    return CacheStatsResponse(**stats)


# ───────────────── Flush Cache ─────────────────


@app.delete("/cache")
async def flush_cache():

    cache.flush()

    return {
        "status": "success",
        "message": "Cache cleared",
    }


# ───────────────── Health Check ─────────────────


@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "vector_store_size": vector_store.size if vector_store else 0,
        "cache_entries": cache.get_stats()["total_entries"]
        if cache
        else 0,
    }


# ───────────────── Root Endpoint ─────────────────


@app.get("/")
async def root():

    return {
        "message": "Semantic Search API running",
        "version": "1.0.0",
    }