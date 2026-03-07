"""
experiments/threshold_experiment.py

Runs the full threshold analysis and saves 4 charts to experiments/plots/.

Usage:
    cd semantic-search
    python experiments/threshold_experiment.py

Requirements:
    - uvicorn must be running: uvicorn api.main:app --port 8000
    - models/ directory must exist (run build.py first)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
from collections import defaultdict
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'

API_BASE = 'http://localhost:8000'
PLOTS_DIR = Path(__file__).parent / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# ── Check API is running ───────────────────────────────────────────────────────
try:
    requests.get(f'{API_BASE}/health', timeout=3)
    print("✅ API is running")
except Exception:
    print("❌ API not running. Start with:")
    print("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
    sys.exit(1)


# ── Load models ───────────────────────────────────────────────────────────────
print("\nLoading models...")
with open('models/cluster_model.pkl', 'rb') as f:
    clusterer = pickle.load(f)
with open('models/documents.pkl', 'rb') as f:
    doc_data = pickle.load(f)
    documents = doc_data['documents']
with open('models/cluster_labels.pkl', 'rb') as f:
    label_data = pickle.load(f)

embeddings      = np.load('models/embeddings.npy')
cluster_labels  = label_data['labels']
cluster_variances = label_data['variances']

print(f"  Documents:  {len(documents)}")
print(f"  Embeddings: {embeddings.shape}")
print(f"  Clusters:   {clusterer.n_clusters}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Adaptive Thresholds
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Analysing adaptive thresholds...")

stats = requests.get(f'{API_BASE}/cache/stats').json()
thresholds = {int(k): v for k, v in stats['adaptive_thresholds'].items()}

print(f"\n{'Cluster':<10} {'Label':<35} {'Threshold':<12} {'Variance'}")
print('-' * 72)
for c in sorted(thresholds.keys()):
    label = cluster_labels.get(c, f'cluster_{c}')
    var   = cluster_variances.get(c, 0)
    note  = ' ← tight' if thresholds[c] > 0.82 else (' ← loose' if thresholds[c] < 0.72 else '')
    print(f"{c:<10} {label:<35} {thresholds[c]:<12.4f} {var:.6f}{note}")

cluster_ids    = sorted(thresholds.keys())
threshold_vals = [thresholds[c]          for c in cluster_ids]
variance_vals  = [cluster_variances.get(c, 0) for c in cluster_ids]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#e74c3c' if t > 0.82 else ('#2ecc71' if t < 0.72 else '#3498db')
          for t in threshold_vals]
axes[0].bar(range(len(cluster_ids)), threshold_vals, color=colors, alpha=0.85, edgecolor='white')
axes[0].axhline(y=0.82, color='black', linestyle='--', linewidth=1.5, label='Base threshold (0.82)')
axes[0].set_xlabel('Cluster ID')
axes[0].set_ylabel('Similarity Threshold')
axes[0].set_title('Adaptive Threshold per Cluster\n(red=tight, blue=medium, green=loose)')
axes[0].set_xticks(range(len(cluster_ids)))
axes[0].set_xticklabels(cluster_ids)
axes[0].set_ylim(0.65, 0.92)
axes[0].legend()

axes[1].scatter(variance_vals, threshold_vals, alpha=0.85, s=80,
                c='#3498db', edgecolors='white', linewidth=0.5)
for c in cluster_ids:
    if thresholds[c] > 0.82 or thresholds[c] < 0.72:
        axes[1].annotate(f'c{c}',
                         (cluster_variances.get(c, 0), thresholds[c]),
                         textcoords='offset points', xytext=(5, 5), fontsize=8)

z = np.polyfit(variance_vals, threshold_vals, 1)
p = np.poly1d(z)
x_line = np.linspace(min(variance_vals), max(variance_vals), 100)
axes[1].plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.2f})')
axes[1].set_xlabel('Intra-cluster Variance')
axes[1].set_ylabel('Similarity Threshold')
axes[1].set_title('Variance → Threshold Relationship\n(higher variance = higher threshold)')
axes[1].legend()

plt.tight_layout()
out = PLOTS_DIR / 'adaptive_thresholds.png'
plt.savefig(out, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Threshold Sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Running threshold sweep...")

from embeddings.embedder import TextEmbedder
embedder = TextEmbedder()

# Semantically equivalent pairs — should be cache hits
TRUE_PAIRS = [
    ("buy a motorcycle",           "purchasing a motorbike"),
    ("gun control legislation",    "firearm regulation debate"),
    ("space shuttle launch",       "NASA rocket mission"),
    ("computer graphics software", "image rendering programs"),
    ("encryption privacy",         "cryptography security"),
    ("used car for sale",          "second hand vehicle purchase"),
    ("baseball game scores",       "MLB match results"),
    ("Windows operating system",   "Microsoft PC software"),
    ("medical treatment advice",   "healthcare recommendations"),
    ("religion faith belief",      "spiritual practice church"),
]

# Semantically different pairs — should NOT be cache hits
FALSE_PAIRS = [
    ("buy a motorcycle",           "gun control legislation"),
    ("space shuttle launch",       "baseball game scores"),
    ("encryption privacy",         "medical treatment advice"),
    ("Windows operating system",   "religion faith belief"),
    ("used car for sale",          "NASA rocket mission"),
    ("computer graphics software", "firearm regulation debate"),
    ("baseball game scores",       "cryptography security"),
    ("gun control legislation",    "image rendering programs"),
]

def cosine_sim(a, b):
    return float(np.dot(a, b))

print("  True pair similarities:")
true_sims = []
for q1, q2 in TRUE_PAIRS:
    sim = cosine_sim(embedder.embed_single(q1), embedder.embed_single(q2))
    true_sims.append(sim)
    print(f"    {sim:.4f}  '{q1}' ↔ '{q2}'")

print("  False pair similarities:")
false_sims = []
for q1, q2 in FALSE_PAIRS:
    sim = cosine_sim(embedder.embed_single(q1), embedder.embed_single(q2))
    false_sims.append(sim)
    print(f"    {sim:.4f}  '{q1}' ↔ '{q2}'  (should NOT match)")

# Sweep
thresholds_sweep = np.arange(0.60, 0.98, 0.02)
true_hit_rates   = [sum(s >= t for s in true_sims)  / len(true_sims)  for t in thresholds_sweep]
false_hit_rates  = [sum(s >= t for s in false_sims) / len(false_sims) for t in thresholds_sweep]

print(f"\n  {'Threshold':<12} {'True Hit Rate':<16} {'False Hit Rate':<16} Notes")
print("  " + "-" * 62)
notes = {0.70: '← too aggressive', 0.78: '← adaptive lower bound',
         0.82: '← base threshold',  0.86: '← adaptive upper', 0.90: '← too conservative'}
for t, thr, fhr in zip(thresholds_sweep, true_hit_rates, false_hit_rates):
    note = notes.get(round(t, 2), '')
    print(f"  {t:<12.2f} {thr:<16.3f} {fhr:<16.3f} {note}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(thresholds_sweep, true_hit_rates,  'g-o', markersize=4, linewidth=2,
             label='True pairs (should hit)')
axes[0].plot(thresholds_sweep, false_hit_rates, 'r-s', markersize=4, linewidth=2,
             label='False pairs (should NOT hit)')
axes[0].axvline(x=0.82, color='black', linestyle='--', alpha=0.7, label='Base threshold (0.82)')
min_t, max_t = min(thresholds.values()), max(thresholds.values())
axes[0].axvspan(min_t, max_t, alpha=0.1, color='blue', label=f'Adaptive range ({min_t:.2f}–{max_t:.2f})')
axes[0].set_xlabel('Similarity Threshold')
axes[0].set_ylabel('Hit Rate')
axes[0].set_title('Cache Hit Rate vs Threshold\n(the core tradeoff)')
axes[0].legend(loc='center left')
axes[0].set_xlim(0.60, 0.97)
axes[0].set_ylim(-0.05, 1.05)

axes[1].hist(true_sims,  bins=15, alpha=0.6, color='green',
             label=f'True pairs (n={len(true_sims)})',  edgecolor='white')
axes[1].hist(false_sims, bins=15, alpha=0.6, color='red',
             label=f'False pairs (n={len(false_sims)})', edgecolor='white')
axes[1].axvline(x=0.82, color='black', linestyle='--', linewidth=2, label='Base threshold')
axes[1].axvspan(min_t, max_t, alpha=0.15, color='blue', label='Adaptive range')
axes[1].set_xlabel('Cosine Similarity')
axes[1].set_ylabel('Count')
axes[1].set_title('Similarity Distribution\n(gap between true/false = system quality)')
axes[1].legend()

plt.tight_layout()
out = PLOTS_DIR / 'threshold_tradeoff.png'
plt.savefig(out, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

true_mean, false_mean = np.mean(true_sims), np.mean(false_sims)
print(f"\n  Mean similarity — true pairs:  {true_mean:.4f}")
print(f"  Mean similarity — false pairs: {false_mean:.4f}")
print(f"  Separation gap:                {true_mean - false_mean:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Boundary Document Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Analysing boundary documents...")

doc_ids       = [d.doc_id for d in documents]
boundary_docs = clusterer.get_boundary_documents(doc_ids, top_n=10)

print("\n  TOP 10 BOUNDARY DOCUMENTS (highest membership entropy)")
print("  " + "=" * 70)
for i, b in enumerate(boundary_docs):
    doc = documents[b['position']]
    print(f"\n  #{i+1}  entropy={b['entropy']:.3f} | category={doc.original_category}")
    for cid, score in b['top_3_clusters']:
        label = cluster_labels.get(cid, f'cluster_{cid}')
        bar   = '█' * int(score * 25)
        print(f"       cluster_{cid:2d} ({label:<28}): {score:.4f}  {bar}")
    print(f"       preview: {doc.text[:150].strip()}")

entropies = clusterer.entropy_per_doc_
max_entropy = np.log2(clusterer.n_clusters)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(entropies, bins=50, color='#3498db', alpha=0.8, edgecolor='white')
axes[0].axvline(x=np.mean(entropies), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(entropies):.3f}')
axes[0].axvline(x=max_entropy, color='gray', linestyle=':', linewidth=1.5,
                label=f'Maximum: {max_entropy:.3f}')
axes[0].set_xlabel('Membership Entropy (bits)')
axes[0].set_ylabel('Number of Documents')
axes[0].set_title('Distribution of Document Entropy\n(lower=clear cluster, higher=boundary doc)')
axes[0].legend()

category_entropies = defaultdict(list)
for i, doc in enumerate(documents):
    category_entropies[doc.original_category].append(entropies[i])

sorted_cats = sorted(category_entropies.items(), key=lambda x: np.mean(x[1]))
cat_names = [c.split('.')[-1][:15] for c, _ in sorted_cats]
cat_means = [np.mean(e) for _, e in sorted_cats]
cat_stds  = [np.std(e)  for _, e in sorted_cats]

axes[1].barh(range(len(cat_names)), cat_means, xerr=cat_stds, alpha=0.8,
             color='#2ecc71', ecolor='gray', capsize=3)
axes[1].set_yticks(range(len(cat_names)))
axes[1].set_yticklabels(cat_names, fontsize=8)
axes[1].set_xlabel('Mean Membership Entropy')
axes[1].set_title('Mean Entropy by Category\n(high entropy = overlaps with others)')
axes[1].axvline(x=np.mean(entropies), color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
out = PLOTS_DIR / 'entropy_analysis.png'
plt.savefig(out, bbox_inches='tight')
plt.close()
print(f"\n  Saved → {out}")

print("\n  Most ambiguous categories:")
for cat, ents in sorted(category_entropies.items(), key=lambda x: np.mean(x[1]), reverse=True)[:5]:
    print(f"    {cat:<35} entropy={np.mean(ents):.3f}")
print("  Most distinct categories:")
for cat, ents in sorted(category_entropies.items(), key=lambda x: np.mean(x[1]))[:5]:
    print(f"    {cat:<35} entropy={np.mean(ents):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Live Cache Performance Demo
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Running live cache demo...")

requests.delete(f'{API_BASE}/cache')
print("  Cache flushed.\n")

demo_queries = [
    "buy a motorcycle",
    "buy a motorcycle",
    "purchasing a motorbike",
    "buying a motorcycle",
    "gun control legislation",
    "firearm regulation debate",
    "space exploration NASA",
    "rocket launch into orbit",
]

results = []
print(f"  {'Status':<8} {'Query':<42} {'Time':>8}  Matched")
print("  " + "-" * 75)
for q in demo_queries:
    r = requests.post(f'{API_BASE}/query', json={'query': q}).json()
    results.append(r)
    status  = '✅ HIT ' if r['cache_hit'] else '❌ MISS'
    matched = f"→ '{r['matched_query'][:25]}' ({r['similarity_score']:.3f})" if r['cache_hit'] else ''
    print(f"  {status}  {q:<42} {r['processing_time_ms']:>6.0f}ms  {matched}")

final_stats = requests.get(f'{API_BASE}/cache/stats').json()
print(f"\n  Final: {final_stats['total_entries']} entries | "
      f"hit_rate={final_stats['hit_rate']:.2%} | "
      f"hits={final_stats['hit_count']} misses={final_stats['miss_count']}")

hit_times  = [r['processing_time_ms'] for r in results if r['cache_hit']]
miss_times = [r['processing_time_ms'] for r in results if not r['cache_hit']]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

avg_miss = np.mean(miss_times) if miss_times else 0
avg_hit  = np.mean(hit_times)  if hit_times  else 0
bars = axes[0].bar(['Cache Miss\n(embed + search)', 'Cache Hit\n(lookup only)'],
                   [avg_miss, avg_hit],
                   color=['#e74c3c', '#2ecc71'], alpha=0.85, width=0.5)
speedup = avg_miss / avg_hit if avg_hit > 0 else 0
axes[0].set_title(f'Response Time: Hit vs Miss\n({speedup:.0f}× faster on cache hit)')
axes[0].set_ylabel('Average Time (ms)')
for i, v in enumerate([avg_miss, avg_hit]):
    axes[0].text(i, v + 3, f'{v:.0f}ms', ha='center', fontweight='bold')

colors_seq = ['#2ecc71' if r['cache_hit'] else '#e74c3c' for r in results]
times_seq  = [r['processing_time_ms'] for r in results]
labels_seq = [(r['query'][:18] + '…') if len(r['query']) > 18 else r['query'] for r in results]

axes[1].bar(range(len(results)), times_seq, color=colors_seq, alpha=0.85, edgecolor='white')
axes[1].set_xticks(range(len(results)))
axes[1].set_xticklabels(labels_seq, rotation=45, ha='right', fontsize=7)
axes[1].set_ylabel('Response Time (ms)')
axes[1].set_title('Query Sequence Timeline\n(green=cache hit, red=cache miss)')
hit_patch  = mpatches.Patch(color='#2ecc71', alpha=0.85, label='Cache Hit')
miss_patch = mpatches.Patch(color='#e74c3c', alpha=0.85, label='Cache Miss')
axes[1].legend(handles=[hit_patch, miss_patch])

plt.tight_layout()
out = PLOTS_DIR / 'cache_performance.png'
plt.savefig(out, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print(f"  Plots saved to: {PLOTS_DIR}/")
print(f"    adaptive_thresholds.png")
print(f"    threshold_tradeoff.png")
print(f"    entropy_analysis.png")
print(f"    cache_performance.png")
print()
print("KEY FINDINGS:")
print(f"  Threshold range:    {min(thresholds.values()):.3f} – {max(thresholds.values()):.3f} (adaptive)")
print(f"  True pair sim mean: {np.mean(true_sims):.4f}")
print(f"  False pair sim mean:{np.mean(false_sims):.4f}")
print(f"  Separation gap:     {np.mean(true_sims) - np.mean(false_sims):.4f}")
print(f"  Cache speedup:      {speedup:.0f}×")
print(f"  Mean doc entropy:   {np.mean(entropies):.3f} / {max_entropy:.3f}")
print("=" * 60)