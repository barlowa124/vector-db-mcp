"""Index registry: create/list/describe/delete indexes under a data dir.

Writes are atomic-ish (tmp dir + rename per index save). The registry is
just a directory of index folders; `VDB_DATA_DIR` or --data-dir selects it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from vdb_mcp.index import VectorIndex


def data_dir() -> Path:
    return Path(os.environ.get("VDB_DATA_DIR", "data/indexes"))


class Store:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else data_dir()
        self._cache: dict[str, VectorIndex] = {}

    def _path(self, name: str) -> Path:
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"invalid index name {name!r}")
        return self.root / name

    def create_index(self, name: str, dimension: int, metric: str = "cosine"):
        p = self._path(name)
        if p.exists():
            raise ValueError(f"index {name!r} already exists")
        idx = VectorIndex(name, dimension, metric)
        idx.save(p)
        self._cache[name] = idx
        return idx.describe()

    def list_indexes(self) -> list[dict]:
        if not self.root.exists():
            return []
        return [VectorIndex.load(p).describe()
                for p in sorted(self.root.iterdir()) if p.is_dir()]

    def get(self, name: str) -> VectorIndex:
        if name not in self._cache:
            p = self._path(name)
            if not p.exists():
                raise KeyError(f"index {name!r} not found")
            self._cache[name] = VectorIndex.load(p)
        return self._cache[name]

    def describe_index(self, name: str) -> dict:
        return self.get(name).describe()

    def delete_index(self, name: str) -> None:
        self._cache.pop(name, None)
        p = self._path(name)
        if not p.exists():
            raise KeyError(f"index {name!r} not found")
        shutil.rmtree(p)

    def describe_index_stats(self, name: str) -> dict:
        return self.get(name).stats()

    # ---- record ops (persist on every mutation) ----
    def upsert(self, index: str, records: list, namespace: str = "") -> dict:
        idx = self.get(index)
        n = idx.upsert(records, namespace)
        idx.save(self._path(index))
        return {"upserted_count": n}

    def query(self, index: str, **kw) -> dict:
        return self.get(index).query(**kw)

    def fetch(self, index: str, ids: list, namespace: str = "") -> dict:
        return self.get(index).fetch(ids, namespace)

    def delete(self, index: str, namespace: str = "", ids=None,
               delete_all: bool = False) -> dict:
        idx = self.get(index)
        n = idx.delete(namespace, ids, delete_all)
        idx.save(self._path(index))
        return {"deleted_count": n}

    def update(self, index: str, namespace: str, id: str, values=None,
               set_metadata=None) -> dict:
        idx = self.get(index)
        if not idx.update(namespace, id, values, set_metadata):
            raise KeyError(f"id {id!r} not in namespace {namespace!r}")
        idx.save(self._path(index))
        return {"updated": id}
