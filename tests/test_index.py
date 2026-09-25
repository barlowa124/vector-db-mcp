import numpy as np
import pytest

from vdb_mcp.index import VectorIndex


def idx(dim=3, metric="cosine"):
    return VectorIndex("t", dim, metric)


def rec(i, v, **meta):
    return {"id": i, "values": v, "metadata": meta}


def test_cosine_ranks_parallel_first():
    i = idx(metric="cosine")
    i.upsert([rec("a", [1, 0, 0]), rec("b", [0.9, 0.1, 0]),
              rec("c", [0, 1, 0])])
    r = i.query(vector=[1, 0, 0], top_k=3)
    assert [m["id"] for m in r["matches"]] == ["a", "b", "c"]
    assert r["matches"][0]["score"] == pytest.approx(1.0)


def test_euclidean_ranks_nearest():
    i = idx(metric="euclidean")
    i.upsert([rec("near", [1, 0, 0]), rec("far", [0, 0, 9])])
    r = i.query(vector=[1, 0, 0])
    assert r["matches"][0]["id"] == "near"
    assert r["matches"][0]["score"] == pytest.approx(0.0)


def test_dotproduct():
    i = idx(metric="dotproduct")
    i.upsert([rec("a", [1, 1, 0]), rec("b", [2, 0, 0])])
    r = i.query(vector=[1, 0, 0])
    assert r["matches"][0]["id"] == "b"  # dot 2 > 1


def test_upsert_overwrites_same_id():
    i = idx()
    i.upsert([rec("a", [1, 0, 0], tag="x")])
    i.upsert([rec("a", [0, 1, 0], tag="y")])
    assert i.stats()["total_vector_count"] == 1
    assert i.fetch(["a"])["vectors"]["a"]["metadata"] == {"tag": "y"}


def test_dim_mismatch_rejected():
    i = idx(dim=4)
    with pytest.raises(ValueError):
        i.upsert([rec("a", [1, 2, 3])])


def test_namespaces_are_isolated():
    i = idx()
    i.upsert([rec("a", [1, 0, 0])], namespace="train")
    i.upsert([rec("b", [0, 1, 0])], namespace="eval")
    assert i.query(vector=[1, 0, 0], namespace="train")[
        "matches"][0]["id"] == "a"
    assert i.query(vector=[1, 0, 0], namespace="eval")[
        "matches"][0]["id"] == "b"
    assert i.query(vector=[1, 0, 0], namespace="")["matches"] == []
    st = i.stats()["namespaces"]
    assert st["train"]["vector_count"] == 1


def test_metadata_filter_limits_matches():
    i = idx()
    i.upsert([rec("a", [1, 0, 0], kind="paper", year=2020),
              rec("b", [0.95, 0.05, 0], kind="dataset", year=2023)])
    r = i.query(vector=[1, 0, 0],
                filter={"kind": {"$eq": "dataset"}, "year": {"$gte": 2022}})
    assert [m["id"] for m in r["matches"]] == ["b"]
    assert r["matches"][0]["metadata"]["year"] == 2023


def test_query_by_id_uses_stored_vector():
    i = idx()
    i.upsert([rec("a", [1, 0, 0]), rec("b", [0, 1, 0])])
    r = i.query(id="a", top_k=2)
    assert r["matches"][0]["id"] == "a"
    with pytest.raises(ValueError):
        i.query(id="missing")


def test_delete_and_delete_all():
    i = idx()
    i.upsert([rec("a", [1, 0, 0]), rec("b", [0, 1, 0])], namespace="n")
    assert i.delete(namespace="n", ids=["a"]) == 1
    assert i.fetch(["a"], "n")["vectors"] == {}
    assert i.delete(namespace="n", delete_all=True) == 1
    assert i.stats()["total_vector_count"] == 0


def test_update_vector_and_metadata():
    i = idx()
    i.upsert([rec("a", [1, 0, 0], keep=1)])
    assert i.update("", "a", values=[0, 1, 0], set_metadata={"new": 2})
    v = i.fetch(["a"])["vectors"]["a"]
    assert v["values"] == [0.0, 1.0, 0.0]
    assert v["metadata"] == {"keep": 1, "new": 2}
    assert not i.update("", "ghost")
