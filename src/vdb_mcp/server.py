"""MCP stdio server exposing the Pinecone-surface tool set.

Tool names mirror Pinecone's API surface (list_indexes, create_index,
describe_index, delete_index, describe_index_stats, upsert, query, fetch,
delete, update). Optional `query_text` embeds locally if the `embed`
extra is installed — Pinecone's integrated-inference equivalent.

Run: VDB_DATA_DIR=... vdb serve    (stdio transport)
"""

from __future__ import annotations

from vdb_mcp.store import Store

store = Store()


def build_server():
    from mcp.server.mcpserver import MCPServer
    m = MCPServer("vector-db-mcp")

    @m.tool()
    def list_indexes() -> list:
        """List all local indexes."""
        return store.list_indexes()

    @m.tool()
    def create_index(name: str, dimension: int, metric: str = "cosine",
                     index_type: str = "flat") -> dict:
        """Create an index. metric: cosine | euclidean | dotproduct.
        index_type: flat (exact) | ivf | hnsw (approximate)."""
        return store.create_index(name, dimension, metric, index_type)

    @m.tool()
    def describe_index(name: str) -> dict:
        """Describe an index (dimension, metric, status)."""
        return store.describe_index(name)

    @m.tool()
    def delete_index(name: str) -> dict:
        store.delete_index(name)
        return {"deleted": name}

    @m.tool()
    def describe_index_stats(name: str) -> dict:
        """Vector counts per namespace."""
        return store.describe_index_stats(name)

    @m.tool()
    def upsert(index: str, records: list, namespace: str = "") -> dict:
        """records: [{id, values: [float], metadata: {...}}]. Existing ids
        are overwritten (Pinecone semantics)."""
        return store.upsert(index, records, namespace)

    @m.tool()
    def query(index: str, namespace: str = "", top_k: int = 10,
              vector: list | None = None, id: str | None = None,
              filter: dict | None = None,
              include_metadata: bool = True) -> dict:
        """Exact top-k by index metric. Provide vector or a stored id.
        filter uses Pinecone operators: $eq $ne $gt $gte $lt $lte $in
        $nin $exists $and $or."""
        return store.query(index, vector=vector, id=id, top_k=top_k,
                           namespace=namespace, filter=filter,
                           include_metadata=include_metadata)

    @m.tool()
    def fetch(index: str, ids: list, namespace: str = "") -> dict:
        """Fetch full records by id."""
        return store.fetch(index, ids, namespace)

    @m.tool()
    def delete(index: str, namespace: str = "", ids: list | None = None,
               delete_all: bool = False) -> dict:
        """Delete by ids, or wipe a namespace with delete_all."""
        return store.delete(index, namespace, ids, delete_all)

    @m.tool()
    def update(index: str, namespace: str, id: str,
               values: list | None = None,
               set_metadata: dict | None = None) -> dict:
        """Update a record's vector and/or merge metadata fields."""
        return store.update(index, namespace, id, values, set_metadata)

    @m.tool()
    def query_text(index: str, text: str, namespace: str = "",
                   top_k: int = 10, filter: dict | None = None) -> dict:
        """Semantic text query: embeds `text` locally and runs `query`.
        Requires the `embed` extra (sentence-transformers E5-small)."""
        try:
            from vdb_mcp.embed import embed_text
        except ImportError:
            raise RuntimeError(
                "query_text needs the embed extra: pip install .[embed]")
        vec = embed_text(text)
        return store.query(index, vector=vec.tolist(), top_k=top_k,
                           namespace=namespace, filter=filter)

    return m


def main() -> None:
    build_server().run()
