"""In-memory vector index with exact (brute-force) search.

Feature-parity scope: Pinecone index semantics — dimension-pinned,
one metric per index, namespaces partition records, upsert overwrites
by id. Difference by design: exact linear scan instead of ANN, so
results are exact rather than approximate, and capacity is RAM.
"""

from __future__ import annotations

import time

import numpy as np

METRICS = ("cosine", "euclidean", "dotproduct")


class VectorIndex:
    def __init__(self, name: str, dimension: int, metric: str = "cosine"):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.name = name
        self.dimension = int(dimension)
        self.metric = metric
        self.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # namespace -> {"ids": [str], "vectors": float32 2D, "meta": [dict]}
        self._ns: dict[str, dict] = {}

    def _empty(self):
        return {"ids": [], "vectors": np.zeros((0, self.dimension),
                                               np.float32), "meta": []}

    def _ns_get(self, ns: str) -> dict:
        return self._ns.setdefault(ns, self._empty())

    def describe(self) -> dict:
        return {"name": self.name, "dimension": self.dimension,
                "metric": self.metric, "created": self.created_utc,
                "host": "local", "status": "ready"}

    def stats(self) -> dict:
        ns = {k: {"vector_count": len(v["ids"])} for k, v in self._ns.items()}
        return {"namespaces": ns,
                "total_vector_count": sum(v["vector_count"]
                                          for v in ns.values()),
                "dimension": self.dimension}

    def upsert(self, records: list[dict], namespace: str = "") -> int:
        d = self._ns_get(namespace)
        ids, vecs, metas = d["ids"], d["vectors"], d["meta"]
        for r in records:
            v = np.asarray(r["values"], dtype=np.float32)
            if v.shape != (self.dimension,):
                raise ValueError(
                    f"record {r.get('id')!r}: expected dim "
                    f"{self.dimension}, got {v.shape}")
            if r["id"] in ids:
                i = ids.index(r["id"])
                vecs[i], metas[i] = v, r.get("metadata", {})
            else:
                ids.append(r["id"])
                vecs = np.vstack([vecs, v[None, :]])
                metas.append(r.get("metadata", {}))
        d["vectors"] = vecs
        return len(records)

    def fetch(self, ids: list[str], namespace: str = "") -> dict:
        d = self._ns.get(namespace) or self._empty()
        out = {}
        for i in ids:
            if i in d["ids"]:
                j = d["ids"].index(i)
                out[i] = {"id": i, "values": d["vectors"][j].tolist(),
                          "metadata": d["meta"][j]}
        return {"vectors": out, "namespace": namespace}

    def _scores(self, vecs: np.ndarray, q: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return -np.linalg.norm(vecs - q, axis=1)
        if self.metric == "dotproduct":
            return vecs @ q
        qn = q / (np.linalg.norm(q) or 1.0)
        vn = vecs / np.where(
            np.linalg.norm(vecs, axis=1, keepdims=True) == 0, 1.0,
            np.linalg.norm(vecs, axis=1, keepdims=True))
        return vn @ qn

    def query(self, vector=None, id=None, top_k: int = 10,
              namespace: str = "", filter: dict | None = None,
              include_metadata: bool = True) -> dict:
        from vdb_mcp.filters import match

        d = self._ns.get(namespace) or self._empty()
        if id is not None:
            if id not in d["ids"]:
                raise ValueError(f"id {id!r} not in namespace")
            vector = d["vectors"][d["ids"].index(id)]
        if vector is None:
            raise ValueError("query requires vector or id")
        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dimension,):
            raise ValueError(f"query dim {q.shape}, index dim "
                             f"{self.dimension}")
        if not d["ids"]:
            return {"matches": [], "namespace": namespace}
        keep = [j for j, m in enumerate(d["meta"]) if match(m, filter)]
        scores = self._scores(d["vectors"][keep], q) if keep else np.array([])
        order = np.argsort(-scores)[:top_k]
        matches = []
        for j in order:
            rec = {"id": d["ids"][keep[j]], "score": float(scores[j])}
            if include_metadata:
                rec["metadata"] = d["meta"][keep[j]]
            matches.append(rec)
        return {"matches": matches, "namespace": namespace}

    def delete(self, namespace: str = "", ids: list[str] | None = None,
               delete_all: bool = False) -> int:
        if namespace not in self._ns:
            return 0
        if delete_all:
            n = len(self._ns[namespace]["ids"])
            del self._ns[namespace]
            return n
        if not ids:
            return 0
        d = self._ns[namespace]
        drop = set(ids) & set(d["ids"])
        if not drop:
            return 0
        keep = [j for j, i in enumerate(d["ids"]) if i not in drop]
        d["ids"] = [d["ids"][j] for j in keep]
        d["vectors"] = d["vectors"][keep]
        d["meta"] = [d["meta"][j] for j in keep]
        return len(drop)

    def update(self, namespace: str, id: str, values=None,
               set_metadata: dict | None = None) -> bool:
        d = self._ns.get(namespace) or self._empty()
        if id not in d["ids"]:
            return False
        j = d["ids"].index(id)
        if values is not None:
            v = np.asarray(values, dtype=np.float32)
            if v.shape != (self.dimension,):
                raise ValueError("dim mismatch")
            d["vectors"][j] = v
        if set_metadata:
            d["meta"][j].update(set_metadata)
        return True

    # ---- persistence ----
    def save(self, path) -> None:
        import json
        from pathlib import Path
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "index.json").write_text(json.dumps({
            "name": self.name, "dimension": self.dimension,
            "metric": self.metric, "created": self.created_utc}, indent=1))
        for ns, d in self._ns.items():
            ns_dir = tmp / (ns if ns else "_default")
            ns_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                ns_dir / "records.npz",
                ids=np.asarray(d["ids"]),
                vectors=d["vectors"],
                meta=np.asarray([json.dumps(m) for m in d["meta"]]))
        if path.exists():
            import shutil
            shutil.rmtree(path)
        tmp.rename(path)

    @classmethod
    def load(cls, path) -> "VectorIndex":
        import json
        from pathlib import Path
        path = Path(path)
        meta = json.loads((path / "index.json").read_text())
        idx = cls(meta["name"], meta["dimension"], meta["metric"])
        idx.created_utc = meta["created"]
        for ns_dir in path.iterdir():
            if not ns_dir.is_dir():
                continue
            z = np.load(ns_dir / "records.npz", allow_pickle=False)
            idx._ns["" if ns_dir.name == "_default" else ns_dir.name] = {
                "ids": z["ids"].tolist(),
                "vectors": z["vectors"].astype(np.float32),
                "meta": [json.loads(m) for m in z["meta"].tolist()],
            }
        return idx
