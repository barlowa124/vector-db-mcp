"""Optional local text embedding for query_text.

E5-small-v2 via transformers; ~33M params, runs on CPU. Query passages
get the 'query: ' prefix per the E5 convention; documents should be
upserted with 'passage: ' embeddings for symmetric retrieval.
"""

import numpy as np

_MODEL = "intfloat/e5-small-v2"
_cache = {}


def _load():
    if "m" not in _cache:
        import torch
        from transformers import AutoModel, AutoTokenizer
        t = AutoTokenizer.from_pretrained(_MODEL)
        m = AutoModel.from_pretrained(_MODEL).eval()
        _cache.update(t=t, m=m, torch=torch)
    return _cache["t"], _cache["m"], _cache["torch"]


def embed_text(text: str) -> np.ndarray:
    t, m, torch = _load()
    enc = t(f"query: {text}", return_tensors="pt", truncation=True,
            max_length=512)
    with torch.no_grad():
        h = m(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    v = (h * mask).sum(1).div(mask.sum(1)).numpy()[0]
    return v / np.linalg.norm(v)


def embed_passage(text: str) -> np.ndarray:
    t, m, torch = _load()
    enc = t(f"passage: {text}", return_tensors="pt", truncation=True,
            max_length=512)
    with torch.no_grad():
        h = m(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    v = (h * mask).sum(1).div(mask.sum(1)).numpy()[0]
    return v / np.linalg.norm(v)
