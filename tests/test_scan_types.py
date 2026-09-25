"""Approximate index types: recall measured against the exact baseline,
not asserted structurally. Thresholds are deliberately loose —
clustered synthetic data should make recall near-perfect."""

import numpy as np
import pytest

from vdb_mcp.index import VectorIndex


def _clustered(n=800, dim=16, n_clusters=8, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim)
    vecs = centers[rng.randint(0, n_clusters, n)] + 0.1 * rng.randn(n, dim)
    return vecs.astype(np.float32)


def _recall(approx: VectorIndex, exact: VectorIndex, queries, k=10):
    hits = 0
    for q in queries:
        truth = {m["id"] for m in exact.query(vector=q, top_k=k)["matches"]}
        got = {m["id"] for m in approx.query(vector=q, top_k=k)["matches"]}
        hits += len(truth & got) / k
    return hits / len(queries)


@pytest.mark.parametrize("index_type", ["ivf", "hnsw"])
def test_approximate_recall(index_type):
    vecs = _clustered()
    records = [{"id": f"v{i}", "values": v.tolist(), "metadata": {}}
               for i, v in enumerate(vecs)]
    exact = VectorIndex("e", 16, "cosine")
    approx = VectorIndex("a", 16, "cosine", index_type)
    exact.upsert(records)
    approx.upsert(records)
    rng = np.random.RandomState(1)
    queries = rng.randn(30, 16).astype(np.float32)
    r = _recall(approx, exact, queries)
    assert r > 0.8, f"{index_type} recall@10 {r:.2f} below 0.8"


def test_flat_is_exact_by_construction():
    vecs = _clustered(200)
    i = VectorIndex("f", 16, "cosine")
    i.upsert([{"id": str(j), "values": v.tolist(), "metadata": {}}
              for j, v in enumerate(vecs)])
    q = np.random.RandomState(2).randn(16)
    got = [m["id"] for m in i.query(vector=q.tolist(), top_k=5)["matches"]]
    scores = vecs.astype(np.float64) @ q / (
        np.linalg.norm(vecs.astype(np.float64), axis=1)
        * np.linalg.norm(q))
    truth = [str(j) for j in np.argsort(-scores)[:5]]
    assert got == truth


def test_index_type_persisted_and_described(tmp_path):
    from vdb_mcp.store import Store
    s = Store(tmp_path)
    s.create_index("h", 4, "cosine", index_type="hnsw")
    assert s.describe_index("h")["index_type"] == "hnsw"
    assert Store(tmp_path).get("h").index_type == "hnsw"


def test_approx_survives_delete(tmp_path):
    vecs = _clustered(300)
    i = VectorIndex("d", 16, "cosine", "hnsw")
    i.upsert([{"id": str(j), "values": v.tolist(), "metadata": {}}
              for j, v in enumerate(vecs)])
    i.query(vector=vecs[0].tolist(), top_k=5)  # build graph
    i.delete(ids=["0", "1", "2"])              # invalidates
    r = i.query(vector=vecs[5].tolist(), top_k=5)
    assert len(r["matches"]) == 5
    assert all(m["id"] not in {"0", "1", "2"} for m in r["matches"])
