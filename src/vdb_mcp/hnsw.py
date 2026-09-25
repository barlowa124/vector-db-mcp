"""Minimal HNSW: hierarchical navigable small-world graph.

Standard construction: geometric level assignment, greedy beam search
per layer during insert, ef-bounded search at query. Compact but real:
multi-level entry point, M-bounded adjacency, ef beam width.
"""

import numpy as np


class HNSW:
    def __init__(self, M: int = 16, ef_construction: int = 64,
                 seed: int = 0):
        self.M = M
        self.M_max = 2 * M  # layer-0 fanout cap
        self.efc = ef_construction
        self.rng = np.random.RandomState(seed)
        self.level_scale = 1.0 / np.log(M)
        self.vectors: list = []
        self.graph: list[dict[int, list]] = []  # per-level adjacency
        self.entry: int | None = None
        self.max_level = -1
        self.dirty = False  # incremental, never rebuilt

    def _level(self) -> int:
        return int(-np.log(self.rng.uniform(1e-9, 1.0)) * self.level_scale)

    def _dist(self, q: np.ndarray, i: int) -> float:
        return float(np.linalg.norm(self.vectors[i] - q))

    def _beam(self, q: np.ndarray, entry: int, level: int,
              ef: int) -> list[int]:
        """Greedy best-first beam at one layer; returns ef nearest nodes."""
        visited = {entry}
        cand = [(self._dist(q, entry), entry)]      # min-heap by dist
        result = [(-self._dist(q, entry), entry)]   # max-heap by -dist
        import heapq
        while cand:
            d, node = heapq.heappop(cand)
            if -result[0][0] < d and len(result) >= ef:
                break
            for nxt in self.graph[level].get(node, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                dn = self._dist(q, nxt)
                if len(result) < ef or dn < -result[0][0]:
                    heapq.heappush(cand, (dn, nxt))
                    heapq.heappush(result, (-dn, nxt))
                    if len(result) > ef:
                        heapq.heappop(result)
        return sorted((-d, i) for d, i in result)

    def _prune(self, node: int, level: int) -> None:
        """Relative-neighborhood selection: keep candidate j only if it is
        closer to `node` than to every already-kept neighbor. Nearest-M
        truncation alone deletes long-range bridge edges and fragments the
        graph into per-cluster components (measured: recall@10 ~0.65,
        only ~1/8 of nodes reachable from the entry point)."""
        adj = self.graph[level].get(node, [])
        cap = self.M_max if level == 0 else self.M
        if len(adj) <= cap:
            return
        v = self.vectors[node]
        ordered = sorted(adj,
                         key=lambda j: np.linalg.norm(v - self.vectors[j]))
        kept = []
        for j in ordered:
            dj = np.linalg.norm(v - self.vectors[j])
            if all(dj < np.linalg.norm(self.vectors[j] - self.vectors[r])
                   for r in kept):
                kept.append(j)
                if len(kept) == cap:
                    break
        self.graph[level][node] = kept

    def add(self, vec: np.ndarray) -> int:
        import heapq
        i = len(self.vectors)
        self.vectors.append(np.asarray(vec, dtype=np.float32))
        level = min(self._level(), 6)
        for l in range(level + 1):
            while len(self.graph) <= l:
                self.graph.append({})
            self.graph[l][i] = []
        if self.entry is None:
            self.entry = i
            self.max_level = level
            return i
        cur = self.entry
        for l in range(self.max_level, level, -1):
            cur = self._beam(self.vectors[i], cur, l, ef=1)[0][1]
        for l in range(min(level, self.max_level), -1, -1):
            cands = self._beam(self.vectors[i], cur, l, self.efc)
            for dist, nb in cands:
                if nb == i:
                    continue
                self.graph[l][nb].append(i)
                self.graph[l][i].append(nb)
                self._prune(nb, l)
            self._prune(i, l)
            # next-layer entry must be another node: the beam now contains
            # i itself (distance zero to itself), which would orphan it
            nxt = [t for t in cands if t[1] != i]
            if nxt:
                cur = nxt[0][1]
        if level > self.max_level:
            self.entry, self.max_level = i, level
        return i

    def search(self, q: np.ndarray, k: int, ef: int | None = None) -> list[int]:
        if self.entry is None:
            return []
        cur = self.entry
        for l in range(self.max_level, 0, -1):
            cur = self._beam(q, cur, l, ef=1)[0][1]
        hits = self._beam(q, cur, 0, ef or max(2 * k, 32))
        return [i for _, i in hits[:k]]
