"""IVF (inverted file) approximate search.

k-means partitions vectors into `nlist` lists; a query scores only the
`nprobe` nearest lists. Rebuilt lazily when dirty and the namespace is
large enough to bother (small namespaces fall back to exact scan).
"""

import numpy as np

MIN_TRAIN = 64  # below this, exact scan is already trivial


def _kmeans(vecs: np.ndarray, k: int, seed: int = 0,
            iters: int = 12) -> np.ndarray:
    rng = np.random.RandomState(seed)
    centroids = vecs[rng.choice(len(vecs), k, replace=False)].copy()
    for _ in range(iters):
        d = ((vecs[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        assign = np.argmin(d, axis=1)
        for j in range(k):
            sel = assign == j
            if sel.any():
                centroids[j] = vecs[sel].mean(0)
    return centroids


class IVF:
    def __init__(self, nlist: int | None = None, nprobe: int = 4,
                 seed: int = 0):
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self.dirty = True
        self.centroids = None
        self.lists: list[list[int]] = []

    def train(self, vecs: np.ndarray) -> None:
        n = len(vecs)
        self.nlist = self.nlist or max(4, int(np.sqrt(n)))
        k = min(self.nlist, max(1, n // 8))
        self.centroids = _kmeans(vecs, k, self.seed)
        d = ((vecs[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        assign = np.argmin(d, axis=1)
        self.lists = [np.where(assign == j)[0].tolist() for j in range(k)]
        self.dirty = False

    def candidates(self, vecs: np.ndarray, q: np.ndarray,
                   nprobe: int | None) -> np.ndarray:
        if self.dirty:
            self.train(vecs)
        probe = self.nprobe if nprobe is None else nprobe
        order = np.argsort(
            ((self.centroids - q) ** 2).sum(1))[:max(1, probe)]
        out = np.concatenate([self.lists[j] for j in order])
        return out.astype(int)
