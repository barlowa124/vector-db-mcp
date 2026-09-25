import numpy as np
import pytest

from vdb_mcp.store import Store


def test_lifecycle(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 4, "cosine")
    s.upsert("docs", [{"id": "a", "values": [1, 0, 0, 0],
                     "metadata": {"k": 1}}], "ns")
    # reload from disk, not cache
    s2 = Store(tmp_path)
    r = s2.query("docs", vector=[1, 0, 0, 0], namespace="ns")
    assert r["matches"][0]["id"] == "a"
    assert s2.describe_index("docs")["metric"] == "cosine"
    s2.delete_index("docs")
    assert s2.list_indexes() == []


def test_duplicate_create_rejected(tmp_path):
    s = Store(tmp_path)
    s.create_index("x", 2)
    with pytest.raises(ValueError):
        s.create_index("x", 2)


def test_unknown_index_raises(tmp_path):
    with pytest.raises(KeyError):
        Store(tmp_path).describe_index("nope")


def test_persistence_roundtrip_values(tmp_path):
    s = Store(tmp_path)
    s.create_index("v", 3)
    s.upsert("v", [{"id": "a", "values": [0.1, 0.2, 0.3],
                    "metadata": {"t": "x"}}])
    idx = Store(tmp_path).get("v")
    v = idx.fetch(["a"])["vectors"]["a"]
    assert v["values"] == pytest.approx([0.1, 0.2, 0.3])
    assert v["metadata"] == {"t": "x"}


def test_namespace_dirs_roundtrip(tmp_path):
    s = Store(tmp_path)
    s.create_index("v", 2)
    s.upsert("v", [{"id": "a", "values": [1, 0], "metadata": {}}], "train")
    s.upsert("v", [{"id": "b", "values": [0, 1], "metadata": {}}], "")
    s2 = Store(tmp_path)
    assert s2.get("v").stats()["namespaces"]["train"]["vector_count"] == 1
    assert s2.query("v", vector=[0, 1])["matches"][0]["id"] == "b"
