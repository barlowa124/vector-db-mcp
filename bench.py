"""Scan-type benchmark: recall@k and query latency vs the exact baseline.

Committed output in results/scan_benchmark.json is the evidence that the
approximate types' trade-off was measured, not asserted.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from vdb_mcp.index import VectorIndex


def clustered(n, dim, n_clusters, seed):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim)
    return (centers[rng.randint(0, n_clusters, n)]
            + 0.1 * rng.randn(n, dim)).astype(np.float32)


def main(out="results/scan_benchmark.json", n=5000, dim=64, nq=100, k=10):
    vecs = clustered(n, dim, 32, seed=0)
    rng = np.random.RandomState(1)
    queries = rng.randn(nq, dim).astype(np.float32)

    exact = VectorIndex("exact", dim, "cosine")
    exact.upsert([{"id": str(i), "values": v.tolist(), "metadata": {}}
                  for i, v in enumerate(vecs)])
    truth = [[m["id"] for m in
              exact.query(vector=q.tolist(), top_k=k)["matches"]]
             for q in queries]

    rows = {}
    for t in ("flat", "ivf", "hnsw"):
        idx = VectorIndex(t, dim, "cosine", t)
        t0 = time.time()
        idx.upsert([{"id": str(i), "values": v.tolist(), "metadata": {}}
                    for i, v in enumerate(vecs)])
        build_s = time.time() - t0
        recall, lat = 0.0, []
        for qi, q in enumerate(queries):
            t0 = time.time()
            got = {m["id"] for m in
                   idx.query(vector=q.tolist(), top_k=k)["matches"]}
            lat.append(time.time() - t0)
            recall += len(set(truth[qi]) & got) / k
        rows[t] = {
            "recall_at_%d" % k: recall / nq,
            "median_query_ms": float(np.median(lat) * 1000),
            "upsert_seconds": build_s,
        }
        print(t, rows[t])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "n": n, "dim": dim, "n_queries": nq, "k": k,
        "note": ("exact flat scan vs approximate types; recall measured "
                 "against flat on the same queries"), "results": rows},
        indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:]))
