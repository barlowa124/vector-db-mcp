# vector-db-mcp

A local vector database exposing the **Pinecone API surface over MCP**
(stdio) — the only differences by design are scale and networking:
records live in `data/indexes/` on local disk and search is an exact
linear scan, not approximate nearest neighbors.

## Why

Agent-consumed retrieval shouldn't require a managed service. The MCP
tool set mirrors Pinecone's so an agent written against this server
ports to the hosted API with a client swap.

## Feature surface

- Indexes: `create_index`, `describe_index`, `list_indexes`,
  `delete_index`, `describe_index_stats`
- Records: `upsert` (id-keyed overwrite), `query`, `fetch`, `delete`
  (ids or `delete_all` per namespace), `update` (vector and/or metadata
  merge)
- Namespaces partition records inside an index
- Metrics: `cosine`, `euclidean`, `dotproduct`
- Metadata filters with Pinecone operators: `$eq $ne $gt $gte $lt $lte
  $in $nin $exists $and $or`
- `query_text`: local E5-small embedding for semantic text queries
  (`pip install .[embed]`); Pinecone's integrated-inference equivalent
- Exact search: results are deterministic and exact (no ANN recall
  error). Persistence is atomic tmp-dir + rename on every mutation.

## Not implemented (the real boundary)

- Distributed/remote serving, replication, sharding
- Approximate NN (HNSW/IVF), sparse/hybrid indexes, rerankers
- Hosted inference integrations

## Run

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e .[dev]
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

MCP server config (stdio):

```json
"vdb": {
  "command": "/path/to/.venv/bin/python",
  "args": ["-m", "vdb_mcp", "serve"],
  "env": {"PYTHONPATH": "src", "VDB_DATA_DIR": "data/indexes"}
}
```

CLI without MCP:

```bash
vdb create docs --dim 384 --metric cosine
vdb upsert docs --ns papers --file records.jsonl   # {id, values, metadata}
vdb query docs --vector-file q.json --ns papers --filter '{"year":{"$gte":2020}}'
```

## Scope

Local, single-process, exact search. Research/education tooling; not a
managed-service replacement at scale, and that is the point — the
interface is honest about where the boundary is.
