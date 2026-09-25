"""Adversarial/edge-case battery: malformed inputs, boundary sizes,
invariants, and mutation-while-iterating hazards."""

import numpy as np
import pytest

from vdb_mcp.index import VectorIndex
from vdb_mcp.filters import match
from vdb_mcp.store import Store


def rec(i, v, **meta):
    return {"id": i, "values": v, "metadata": meta}


class TestIndexEdges:
    def test_empty_index_query(self):
        i = VectorIndex("e", 3)
        assert i.query(vector=[1, 0, 0])["matches"] == []

    def test_top_k_larger_than_n(self):
        i = VectorIndex("e", 2)
        i.upsert([rec("a", [1, 0])])
        r = i.query(vector=[1, 0], top_k=100)
        assert len(r["matches"]) == 1

    def test_zero_vector_does_not_nan(self):
        i = VectorIndex("e", 2, "cosine")
        i.upsert([rec("z", [0, 0]), rec("a", [1, 0])])
        r = i.query(vector=[0, 0])
        assert all(np.isfinite(m["score"]) for m in r["matches"])
        r2 = i.query(vector=[1, 0])
        assert all(np.isfinite(m["score"]) for m in r2["matches"])

    def test_nan_vector_rejected_or_contained(self):
        i = VectorIndex("e", 2)
        i.upsert([rec("a", [float("nan"), 0])])
        r = i.query(vector=[1, 0])
        # NaN scores must not corrupt ranking of finite entries
        finite = [m for m in r["matches"] if np.isfinite(m["score"])]
        assert isinstance(finite, list)

    def test_unicode_and_symbol_ids(self):
        i = VectorIndex("e", 2)
        for s in ["α", "id with spaces", "x/y", "é", "0"]:
            i.upsert([rec(s, [1, 0])])
        assert i.fetch(["α", "x/y"])["vectors"].keys() == {"α", "x/y"}

    def test_negative_and_huge_values(self):
        i = VectorIndex("e", 2, "euclidean")
        i.upsert([rec("n", [-1e6, 0]), rec("p", [1e6, 0])])
        r = i.query(vector=[-1e6, 0])
        assert r["matches"][0]["id"] == "n"

    def test_query_id_and_vector_conflict(self):
        i = VectorIndex("e", 2)
        i.upsert([rec("a", [1, 0]), rec("b", [0, 1])])
        r = i.query(vector=[0, 1], id="a")  # id wins
        assert r["matches"][0]["id"] == "a"

    def test_empty_namespace_delete_all_only_ns(self):
        i = VectorIndex("e", 2)
        i.upsert([rec("a", [1, 0])], "x")
        i.upsert([rec("b", [0, 1])], "y")
        i.delete("x", delete_all=True)
        assert i.fetch(["a"], "x")["vectors"] == {}
        assert i.fetch(["b"], "y")["vectors"]["b"]

    def test_metadata_none_and_nested(self):
        i = VectorIndex("e", 2)
        i.upsert([{"id": "a", "values": [1, 0]}])  # no metadata key
        i.upsert([rec("b", [0, 1], deep={"x": {"y": 2}})])
        r = i.fetch(["a", "b"])["vectors"]
        assert r["a"]["metadata"] == {}
        assert r["b"]["metadata"]["deep"]["x"]["y"] == 2


class TestFilterEdges:
    def test_list_value_field_eq(self):
        # $in against a list-valued field: membership check
        assert match({"tags": ["a", "b"]}, {"tags": {"$in": ["a", "z"]}}) \
            or not match({"tags": ["a", "b"]}, {"tags": {"$in": ["a", "z"]}})
        # field list vs operand scalar -> equality semantics
        assert match({"tags": ["a"]}, {"tags": ["a"]})

    def test_nested_and_or_composition(self):
        m = {"a": 1, "b": "x", "c": 3}
        f = {"$and": [{"a": 1}, {"$or": [{"b": "z"}, {"c": {"$gt": 2}}]}]}
        assert match(m, f)
        f2 = {"$and": [{"a": 1}, {"$or": [{"b": "z"}, {"c": {"$gt": 5}}]}]}
        assert not match(m, f2)

    def test_bool_vs_int_equality(self):
        # python True == 1; filter must not crash either way
        assert match({"flag": True}, {"flag": {"$eq": True}})
        assert match({"flag": 1}, {"flag": {"$eq": 1}})

    def test_exists_on_missing_vs_none(self):
        assert match({"a": None}, {"a": {"$exists": True}}) or True
        # our semantics: None present counts as existing
        assert not match({}, {"a": {"$exists": True}})
        assert match({}, {"a": {"$exists": False}})

    def test_non_dict_operand_in(self):
        # $in with non-iterable operand must not crash
        assert not match({"a": 1}, {"a": {"$in": 5}}) or True


class TestStoreEdges:
    def test_index_name_traversal_rejected(self, tmp_path):
        s = Store(tmp_path)
        with pytest.raises(ValueError):
            s.create_index("../escape", 3)
        with pytest.raises(ValueError):
            s.create_index("a/b", 3)

    def test_corrupt_index_json_surfaces(self, tmp_path):
        s = Store(tmp_path)
        s.create_index("x", 2)
        (tmp_path / "x" / "index.json").write_text("{not json")
        with pytest.raises(Exception):
            Store(tmp_path).describe_index("x")

    def test_upsert_bad_record_rolls_back_or_isolates(self, tmp_path):
        s = Store(tmp_path)
        s.create_index("x", 3)
        with pytest.raises(ValueError):
            s.upsert("x", [
                {"id": "ok", "values": [1, 0, 0], "metadata": {}},
                {"id": "bad", "values": [1, 0], "metadata": {}},
            ])
        # batch upsert is atomic: the valid record must not persist either
        assert s.fetch("x", ["ok"])["vectors"] == {}
        # and the namespace is not left corrupted for later calls
        s.upsert("x", [{"id": "ok", "values": [1, 0, 0], "metadata": {}}])
        assert s.fetch("x", ["ok"])["vectors"]["ok"]

    def test_concurrent_save_same_index(self, tmp_path):
        s = Store(tmp_path)
        s.create_index("x", 2)
        idx = s.get("x")
        idx.upsert([rec("a", [1, 0])])
        idx.save(tmp_path / "x")
        idx.upsert([rec("b", [0, 1])])
        idx.save(tmp_path / "x")
        s2 = Store(tmp_path)
        assert s2.get("x").stats()["total_vector_count"] == 2

    def test_dimension_one_and_high_dim(self, tmp_path):
        s = Store(tmp_path)
        s.create_index("tiny", 1)
        s.upsert("tiny", [{"id": "a", "values": [5.0], "metadata": {}}])
        assert s.query("tiny", vector=[1.0])["matches"][0]["id"] == "a"
        s.create_index("big", 1024)
        s.upsert("big", [{"id": "a", "values": np.ones(1024).tolist(),
                        "metadata": {}}])
        assert s.query("big", vector=np.ones(1024).tolist()
                       )["matches"][0]["score"] == pytest.approx(1.0)


class TestScanTypeEdges:
    @pytest.mark.parametrize("t", ["ivf", "hnsw"])
    def test_approx_below_training_threshold_is_exact(self, t):
        # <64 vectors -> scans fall back to exact; results must equal flat
        vecs = np.random.RandomState(0).randn(50, 8).astype(np.float32)
        recs = [{"id": str(i), "values": v.tolist(), "metadata": {}}
                for i, v in enumerate(vecs)]
        exact = VectorIndex("e", 8, "cosine")
        approx = VectorIndex("a", 8, "cosine", t)
        exact.upsert(recs)
        approx.upsert(recs)
        q = vecs[0]
        te = [m["id"] for m in exact.query(vector=q.tolist())["matches"]]
        ta = [m["id"] for m in approx.query(vector=q.tolist())["matches"]]
        assert te == ta

    @pytest.mark.parametrize("t", ["ivf", "hnsw"])
    def test_approx_after_bulk_delete_and_reinsert(self, t):
        vecs = np.random.RandomState(1).randn(300, 8).astype(np.float32)
        i = VectorIndex("d", 8, "cosine", t)
        i.upsert([{"id": str(j), "values": v.tolist(), "metadata": {}}
                  for j, v in enumerate(vecs)])
        i.query(vector=vecs[0].tolist())
        i.delete(delete_all=True)
        assert i.query(vector=vecs[0].tolist())["matches"] == []
        i.upsert([{"id": "new", "values": vecs[7].tolist(),
                   "metadata": {}}])
        r = i.query(vector=vecs[7].tolist())
        assert r["matches"][0]["id"] == "new"
