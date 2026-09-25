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
INDEX_TYPES = ("flat", "ivf", "hnsw")


class VectorIndex:
    def __init__(self, name: str, dimension: int, metric: str = "cosine",
                 index_type: str = "flat"):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if index_type not in INDEX_TYPES:
            raise ValueError(f"index_type must be one of {INDEX_TYPES}")
        self.name = name
        self.dimension = int(dimension)
        self.metric = metric
        self.index_type = index_type
        self.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # namespace -> {"ids": [str], "vectors": float32 2D, "meta": [dict],
        #               "scan": IVF|HNSW|None (derived, rebuilt on load)}
        self._ns: dict[str, dict] = {}

    def _empty(self):
        return {"ids": [], "vectors": np.zeros((0, self.dimension),
                                               np.float32), "meta": [],
                "scan": self._new_scan()}

    def _new_scan(self):
        if self.index_type == "ivf":
            from vdb_mcp.ivf import IVF
            return IVF()
        if self.index_type == "hnsw":
            from vdb_mcp.hnsw import HNSW
            # ef_construction is the quality knob: 200 reaches recall@10
            # ~0.95 on the bench corpus vs ~0.65 at the naive default
            return HNSW(M=32, ef_construction=200)
        return None

    def _ns_get(self, ns: str) -> dict:
        return self._ns.setdefault(ns, self._empty())

    def describe(self) -> dict:
        return {"name": self.name, "dimension": self.dimension,
                "metric": self.metric, "index_type": self.index_type,
                "created": self.created_utc,
                "host": "local", "status": "ready"}

    def stats(self) -> dict:
        ns = {k: {"vector_count": len(v["ids"])} for k, v in self._ns.items()}
        return {"namespaces": ns,
                "total_vector_count": sum(v["vector_count"]
                                          for v in ns.values()),
                "dimension": self.dimension}

    def upsert(self, records: list[dict], namespace: str = "") -> int:
        # validate the whole batch first: a bad record mid-batch must not
        # leave ids/meta appended while vectors are not (non-atomic writes
        # corrupt the namespace for every later call)
        parsed = []
        for r in records:
            v = np.asarray(r["values"], dtype=np.float32)
            if v.shape != (self.dimension,):
                raise ValueError(
                    f"record {r.get('id')!r}: expected dim "
                    f"{self.dimension}, got {v.shape}")
            parsed.append((r["id"], v, r.get("metadata", {})))
        d = self._ns_get(namespace)
        ids, vecs, metas = d["ids"], d["vectors"], d["meta"]
        for rid, v, meta in parsed:
            if rid in ids:
                i = ids.index(rid)
                vecs[i], metas[i] = v, meta
                self._invalidate(d)
            else:
                ids.append(rid)
                vecs = np.vstack([vecs, v[None, :]])
                metas.append(meta)
                scan = d["scan"]
                if scan is not None and self.index_type == "hnsw":
                    scan.add(self._scan_space(v[None, :])[0])
                elif scan is not None:
                    scan.dirty = True
        d["vectors"] = vecs
        return len(records)

    def _scan_space(self, vecs: np.ndarray) -> np.ndarray:
        """Vector space the structure navigates in. Cosine indexes search
        normalized copies so euclidean graph/list order agrees with the
        cosine ranking."""
        if self.metric != "cosine":
            return vecs
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(n == 0, 1.0, n)

    def _candidates(self, d: dict, q: np.ndarray, top_k: int) -> list[int]:
        """Candidate row indices: exact for flat, approximate for
        ivf/hnsw. Small namespaces always scan exactly."""
        n = len(d["ids"])
        scan = d["scan"]
        if scan is None or n < 64:
            return list(range(n))
        vecs = self._scan_space(d["vectors"])
        qv = self._scan_space(q[None, :])[0]
        if self.index_type == "ivf":
            return scan.candidates(vecs, qv, None).tolist()
        if scan.dirty or len(scan.vectors) != n:
            scan = self._new_scan()
            for v in vecs:
                scan.add(v)
            d["scan"] = scan
        return scan.search(qv, k=min(n, max(top_k * 8, 64)),
                           ef=max(160, top_k * 16))

    @staticmethod
    def _invalidate(d: dict) -> None:
        if d["scan"] is not None:
            d["scan"].dirty = True

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
        cand = self._candidates(d, q, top_k)
        keep = [j for j in cand if match(d["meta"][j], filter)]
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
        self._invalidate(d)
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
        self._invalidate(d)
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
            "metric": self.metric, "index_type": self.index_type,
            "created": self.created_utc}, indent=1))
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
        idx = cls(meta["name"], meta["dimension"], meta["metric"],
                  meta.get("index_type", "flat"))
        idx.created_utc = meta["created"]
        for ns_dir in path.iterdir():
            if not ns_dir.is_dir():
                continue
            z = np.load(ns_dir / "records.npz", allow_pickle=False)
            d = idx._empty()
            d.update({
                "ids": z["ids"].tolist(),
                "vectors": z["vectors"].astype(np.float32),
                "meta": [json.loads(m) for m in z["meta"].tolist()],
            })
            idx._ns["" if ns_dir.name == "_default" else ns_dir.name] = d
        return idx
