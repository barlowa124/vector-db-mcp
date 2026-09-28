"""Index registry: create/list/describe/delete indexes under a data dir.

Writes are atomic-ish (tmp dir + rename per index save). The registry is
just a directory of index folders; `VDB_DATA_DIR` or --data-dir selects it.
"""

from __future__ import annotations

import functools
import os
import shutil
import threading
from pathlib import Path, PureWindowsPath

from vdb_mcp.index import VectorIndex

_STORE_LOCK = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _STORE_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def data_dir() -> Path:
    return Path(os.environ.get("VDB_DATA_DIR", "data/indexes"))


class Store:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else data_dir()
        self._cache: dict[str, VectorIndex] = {}

    def _path(self, name: str) -> Path:
        if (not isinstance(name, str) or not name or name.startswith('.')
                or name.endswith('.tmp') or name.rstrip(' .') != name
                or any(c in name for c in '/\\:\x00<>"|?*')
                or PureWindowsPath(name).drive or PureWindowsPath(name).is_reserved()):
            raise ValueError(f'invalid index name {name!r}')
        path = self.root / name
        if path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise ValueError(f'index path escapes root: {name!r}')
        return path

    @_locked
    def create_index(self, name: str, dimension: int,
                     metric: str = "cosine", index_type: str = "flat"):
        p = self._path(name)
        VectorIndex.recover(p)
        if p.exists():
            raise ValueError(f"index {name!r} already exists")
        idx = VectorIndex(name, dimension, metric, index_type)
        idx.save(p)
        self._cache[name] = idx
        return idx.describe()

    @_locked
    def list_indexes(self) -> list[dict]:
        if not self.root.exists():
            return []
        for p in self.root.iterdir():
            name = p.name
            if name.startswith('.') and name.endswith('.backup'):
                try:
                    VectorIndex.recover(self._path(name[1:-len('.backup')]))
                except ValueError:
                    pass
        out = []
        for p in sorted(self.root.iterdir()):
            if (not p.is_dir() or p.name.startswith('.')
                    or p.name.endswith('.tmp')):
                continue
            try:
                self._path(p.name)
            except ValueError:
                continue
            out.append(VectorIndex.load(p).describe())
        return out

    @_locked
    def get(self, name: str) -> VectorIndex:
        p = self._path(name)
        VectorIndex.recover(p)
        if name not in self._cache:
            if not p.exists():
                raise KeyError(f"index {name!r} not found")
            self._cache[name] = VectorIndex.load(p)
        return self._cache[name]

    @_locked
    def describe_index(self, name: str) -> dict:
        return self.get(name).describe()

    @_locked
    def delete_index(self, name: str) -> None:
        self._cache.pop(name, None)
        p = self._path(name)
        VectorIndex.recover(p)
        if not p.exists():
            raise KeyError(f"index {name!r} not found")
        backup = p.with_name(f'.{p.name}.backup')
        if backup.exists():
            shutil.rmtree(backup)
        shutil.rmtree(p)

    @_locked
    def describe_index_stats(self, name: str) -> dict:
        return self.get(name).stats()

    def _write(self, index: str, operation):
        idx = self.get(index)
        try:
            result = operation(idx)
            idx.save(self._path(index))
            return result
        except BaseException:
            self._cache.pop(index, None)
            raise

    # ---- record ops (persist on every mutation) ----
    @_locked
    def upsert(self, index: str, records: list, namespace: str = "") -> dict:
        n = self._write(index, lambda idx: idx.upsert(records, namespace))
        return {"upserted_count": n}

    @_locked
    def query(self, index: str, **kw) -> dict:
        return self.get(index).query(**kw)

    @_locked
    def fetch(self, index: str, ids: list, namespace: str = "") -> dict:
        return self.get(index).fetch(ids, namespace)

    @_locked
    def delete(self, index: str, namespace: str = "", ids=None,
               delete_all: bool = False) -> dict:
        n = self._write(
            index, lambda idx: idx.delete(namespace, ids, delete_all))
        return {"deleted_count": n}

    @_locked
    def update(self, index: str, namespace: str, id: str, values=None,
               set_metadata=None) -> dict:
        def op(idx):
            if not idx.update(namespace, id, values, set_metadata):
                raise KeyError(f"id {id!r} not in namespace {namespace!r}")
        self._write(index, op)
        return {"updated": id}
