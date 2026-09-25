"""The MCP tool layer is exercised directly: FastMCP tools are regular
callables via .fn(), which keeps dispatch tests free of a live stdio
loop."""

import pytest

from vdb_mcp import server as srv


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from vdb_mcp.store import Store
    s = Store(tmp_path)
    monkeypatch.setattr(srv, "store", s)
    return s


def _tool(store, name):
    m = srv.build_server()
    return {n: m._tool_manager.get_tool(n).fn for n in m._tool_manager._tools}[name]


def test_tools_end_to_end(store):
    create = _tool(store, "create_index")
    upsert = _tool(store, "upsert")
    query = _tool(store, "query")
    fetch = _tool(store, "fetch")
    delete = _tool(store, "delete")
    stats = _tool(store, "describe_index_stats")

    assert create("docs", 3, "cosine")["status"] == "ready"
    assert upsert("docs", [
        {"id": "a", "values": [1, 0, 0], "metadata": {"kind": "x"}},
        {"id": "b", "values": [0, 1, 0], "metadata": {"kind": "y"}},
    ]) == {"upserted_count": 2}
    r = query("docs", vector=[1, 0, 0], top_k=2)
    assert r["matches"][0]["id"] == "a"
    assert fetch("docs", ["b"])["vectors"]["b"]["values"] == [0, 1, 0]
    assert stats("docs")["total_vector_count"] == 2
    assert delete("docs", ids=["a"]) == {"deleted_count": 1}
    assert stats("docs")["total_vector_count"] == 1


def test_query_text_requires_embed_extra(store):
    qt = _tool(store, "query_text")
    store.create_index("d", 2)
    try:
        qt("d", "hello")
    except RuntimeError as e:
        assert "embed" in str(e)
    except ImportError:
        pytest.skip("embed extra installed")
