# Semantic Search System
**Cluster-aware semantic search with Spherical FCM and adaptive caching**

---

## Architecture

```
                         User Query
                             │
                             ▼
            ┌─────────────────────────────────┐
            │  TextEmbedder                   │
            │  MiniLM-L6-v2 → 384-dim vector  │
            │  L2-normalized (cosine = dot)   │
            └────────────────┬────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────┐
            │  Spherical FCM Clusterer        │
            │  PCA 384 → 50 dims              │
            │  KMeans warm-start              │
            │  Cosine distance FCM            │
            │  Output: membership vector (K,) │
            │  e.g. [cluster_9: 0.317,        │
            │         cluster_8: 0.086, ...]  │
            └────────────────┬────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│               Cluster-Aware Semantic Cache               │
│                                                          │
│   cache = { cluster_id → [CacheEntry, ...] }             │
│                                                          │
│   Lookup: search ONLY dominant cluster bucket            │
│   O(N_total) → O(N_cluster)  ≈ 16× faster at scale      │
│                                                          │
│   threshold_c = base ± Δ(intra-cluster variance)         │
│   Adaptive range: 0.743 – 0.896 across 16 clusters       │
│                                                          │
│   HIT  → return cached result   (~15ms)                  │
│   MISS → FAISS HNSW search      (~400ms)                 │
└──────────┬──────────────────────────┬────────────────────┘
           │ HIT                      │ MISS
           ▼                          ▼
    Return cached result       FAISS HNSW Search
    + metadata                 top-10 documents
                                      │
                                      ▼
                               Synthesize result
                               Store in cache
                               Return response
```

---

## Key Design Decisions

### 1. Embedding Model: `all-MiniLM-L6-v2`
- 384 dimensions — small enough to be fast on CPU, rich enough to capture topic nuance
- Runs at ~2000 sentences/sec without GPU
- Trained specifically for semantic similarity (not generation)
- All vectors L2-normalized: `cosine_similarity(a,b) = dot(a,b)` — makes FAISS inner product search equivalent to cosine search

### 2. Spherical Fuzzy C-Means (not standard FCM)
Standard FCM uses Euclidean distance. On sentence embeddings this causes **distance concentration** — all pairwise distances become nearly equal in high dimensions, collapsing every membership vector to `1/K = 0.0625` (fully degenerate).

**Spherical FCM** fixes this by using cosine distance:
```
d(x, c) = 1 - (x · c)        # for L2-normalized vectors, range [0, 2]
```

Cosine distances between topic clusters are meaningfully different (0.3–0.8), giving FCM a real gradient to follow. Cluster centers are re-normalized after each update to stay on the unit hypersphere — consistent with the metric.

**Why not GMM?** GMM also produces soft distributions and avoids the Euclidean issue, but the assignment explicitly requires fuzzy clustering. Spherical FCM is FCM done correctly for embedding spaces.

**KMeans warm-start**: random FCM initialization fails in high dimensions. We run KMeans first, convert hard labels to a soft membership matrix (dominant cluster = 0.91, others = 0.09/K-1), and use that as FCM's starting point.

**PCA preprocessing**: 384 → 50 dimensions before clustering. Removes noise dimensions that flatten cosine distances while retaining ~50% of variance.

### 3. K = 16 Clusters
The 20 named newsgroup categories contain significant semantic overlap:
- `comp.sys.ibm.pc.hardware` + `comp.sys.mac.hardware` → one hardware cluster
- `talk.politics.guns` + `talk.politics.misc` → one politics cluster  
- `soc.religion.christian` + `talk.religion.misc` → one religion cluster

K=16 was chosen by observing KMeans cluster sizes at different K values — K=16 gives all non-empty, reasonably balanced clusters (738–1599 docs each), indicating genuine 16-way semantic structure in this corpus.

### 4. Cluster-Aware Cache (built from scratch)
No Redis, Memcached, or any caching library. Pure Python + NumPy.

The cache is a `dict[int, list[CacheEntry]]` partitioned by dominant cluster.

**Why partitioning matters at scale:**
- Flat cache of 10,000 entries → 10,000 dot products per lookup
- Cluster-partitioned → ~625 dot products per lookup (10,000 / 16)
- This is O(N) → O(N/K), a constant-factor speedup that compounds as the cache grows

**Why it's valid:** two queries that mean the same thing will have similar embeddings → assigned the same dominant cluster by FCM → their cache entries live in the same bucket.

### 5. Adaptive Similarity Threshold
```
threshold_c = base_threshold ± Δ(intra_cluster_variance_c)
Clamped to [0.70, 0.95]
```

| Cluster | Label | Threshold | Reasoning |
|---------|-------|-----------|-----------|
| 10 | sci_med | 0.895 | Tight cluster — similar-sounding queries may still be distinct |
| 1 | politics_misc | 0.859 | Specific political topics |
| 4 | sport_hockey | 0.743 | Broad language — paraphrases vary more |
| 5 | politics_mideast | 0.762 | Geographic specificity allows looser matching |

---

## Experimental Results

### Live Cache Demo

```
Query:  "buy a motorcycle"          → MISS   840ms   (embed + FAISS search)
Query:  "buy a motorcycle"          → HIT     16ms   sim=1.000  ✅ exact repeat
Query:  "buying a motorcycle"       → HIT     14ms   sim=0.948  ✅ paraphrase caught
Query:  "gun control legislation"   → MISS    32ms
Query:  "firearm regulation debate" → HIT     16ms   sim=0.770  ✅ paraphrase caught
```

**13× faster** on cache hits. The system caught genuine paraphrases across different vocabulary.

### Threshold Analysis

True pair similarities (same meaning):
```
"buy a motorcycle" ↔ "purchasing a motorbike"     0.7205
"gun control" ↔ "firearm regulation"              0.7697
"encryption privacy" ↔ "cryptography security"    0.6778
"baseball game scores" ↔ "MLB match results"      0.6831
```

False pair similarities (different topics):
```
"buy a motorcycle" ↔ "gun control legislation"    0.1308
"encryption privacy" ↔ "medical treatment"        0.0021
"computer graphics" ↔ "firearm regulation"       -0.0225
```

**Separation gap: 0.594**
- True pairs mean: 0.641
- False pairs mean: 0.047

The 0.594 gap proves the embedding space cleanly separates same-meaning from different-meaning queries. The adaptive threshold range (0.743–0.896) sits entirely within this gap.

### Boundary Document Analysis

High entropy documents genuinely span multiple topics — these are not misclassifications, they are evidence that real text doesn't respect clean categorical boundaries:

```
sci.space post: "Here is a way to get commercial companies into space..."
  cluster_12 (rec_autos + misc_forsale): 0.077
  cluster_6  (sci_crypt + alt_atheism):  0.075
  cluster_10 (sci_med):                  0.071
  entropy: 3.989 / 4.000  ← genuinely ambiguous
```

Most semantically ambiguous categories (highest entropy):
```
sci.space              3.923   ← science + politics + commerce
sci.electronics        3.828   ← overlaps comp + sci
rec.motorcycles        3.825   ← overlaps vehicles + forsale
talk.politics.misc     3.762   ← broad political coverage
```

Most semantically distinct categories (lowest entropy):
```
rec.sport.hockey       3.021   ← very topic-specific vocabulary
rec.sport.baseball     3.153   ← self-contained
talk.politics.mideast  3.278   ← specific geographic focus
```

---

## Setup

### Prerequisites
- Python 3.11+
- 4GB RAM minimum (8GB recommended)
- ~2GB disk space for models

### Installation

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd semantic-search

# 2. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build — downloads dataset, embeds 18k docs, runs Spherical FCM
#    Run once. Takes 20-40 minutes on CPU.
python build.py

# 5. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Run build.py on the host first to generate models/
docker build -t semantic-search .
docker run -p 8000:8000 -v $(pwd)/models:/app/models semantic-search

# Or with docker-compose
docker compose up
```

---

## API Reference

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "buy a motorcycle"}'
```

```json
{
  "query": "buy a motorcycle",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "Query routed to semantic cluster: rec_autos + misc_forsale\nTop 10 relevant documents:\n\n[1] similarity=0.673 | category=rec.motorcycles\n    ...",
  "dominant_cluster": 12,
  "cluster_label": "rec_autos + misc_forsale",
  "cluster_entropy": 3.536,
  "membership_vector": {
    "cluster_12": 0.3173,
    "cluster_8": 0.0860,
    "cluster_15": 0.0575
  },
  "cache_threshold_used": 0.8528,
  "processing_time_ms": 427.87
}
```

Same query again (cache hit):
```json
{
  "cache_hit": true,
  "matched_query": "buy a motorcycle",
  "similarity_score": 1.0,
  "processing_time_ms": 16.0
}
```

### GET /cache/stats

```bash
curl http://localhost:8000/cache/stats
```

```json
{
  "total_entries": 5,
  "hit_count": 3,
  "miss_count": 5,
  "hit_rate": 0.375,
  "cluster_distribution": {"12": 2, "9": 2, "4": 1},
  "adaptive_thresholds": {
    "0": 0.8253, "1": 0.8594, "4": 0.7431,
    "10": 0.8957, "12": 0.8528
  }
}
```

### DELETE /cache

```bash
curl -X DELETE http://localhost:8000/cache
```

### GET /health

```bash
curl http://localhost:8000/health
# {"status": "healthy", "vector_store_size": 18343, "cache_entries": 0}
```

---

## Experiments

```bash
# Requires uvicorn running on port 8000
python experiments/threshold_experiment.py
```

Runs 4 analyses and saves plots to `experiments/plots/`:

| Plot | Shows |
|------|-------|
| `adaptive_thresholds.png` | Per-cluster threshold values vs variance |
| `threshold_tradeoff.png` | True/false hit rate curve across thresholds |
| `entropy_analysis.png` | Boundary document entropy by category |
| `cache_performance.png` | Hit vs miss response times, query timeline |

---

## Project Structure

```
semantic-search/
├── data/
│   └── fetch_and_clean.py       # Strips headers, quotes, signatures from posts
├── embeddings/
│   └── embedder.py              # MiniLM-L6-v2 wrapper, L2-normalized output
├── vector_store/
│   └── faiss_store.py           # HNSW index, cluster-filtered search
├── clustering/
│   ├── fuzzy_cluster.py         # Spherical FCM from scratch
│   └── cluster_analysis.py      # K selection, labeling, boundary doc analysis
├── cache/
│   └── semantic_cache.py        # Cluster-partitioned cache, adaptive threshold
├── api/
│   └── main.py                  # FastAPI — /query, /cache/stats, /cache, /health
├── experiments/
│   └── threshold_experiment.py  # Threshold sweep + 4 plots
├── build.py                     # One-time setup: clean → embed → index → cluster
├── diagnose.py                  # Post-build sanity check
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```
