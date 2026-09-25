import pytest

from vdb_mcp.filters import match


def test_scalar_is_implicit_eq():
    assert match({"a": 1}, {"a": 1})
    assert not match({"a": 1}, {"a": 2})
    assert not match({}, {"a": 1})


def test_comparison_ops():
    m = {"year": 2020, "w": 0.5}
    assert match(m, {"year": {"$gte": 2020, "$lte": 2021}})
    assert not match(m, {"year": {"$lt": 2020}})
    assert match(m, {"w": {"$gt": 0.4}})


def test_in_nin_exists():
    m = {"tag": "a", "x": 1}
    assert match(m, {"tag": {"$in": ["a", "b"]}})
    assert not match(m, {"tag": {"$nin": ["a"]}})
    assert match(m, {"x": {"$exists": True}, "y": {"$exists": False}})


def test_and_or():
    m = {"a": 1, "b": 2}
    assert match(m, {"$and": [{"a": 1}, {"b": 2}]})
    assert not match(m, {"$and": [{"a": 1}, {"b": 3}]})
    assert match(m, {"$or": [{"a": 9}, {"b": 2}]})


def test_type_mismatch_comparison_is_false():
    assert not match({"a": "text"}, {"a": {"$gt": 3}})


def test_unknown_operator_raises():
    with pytest.raises(ValueError):
        match({"a": 1}, {"a": {"$bogus": 1}})


def test_empty_and_none_filter_match_everything():
    assert match({"a": 1}, None) and match({"a": 1}, {})
