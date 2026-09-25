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
- Persistence is atomic tmp-dir + rename on every mutation.

## Scan types (`index_type` at `create_index`)

| Type | Search | Recall@10 | Query | Build |
|---|---|---:|---:|---:|
| `flat` | exact linear scan | 1.000 | 1.51 ms | 0.21 s |
| `ivf` | k-means lists, `nprobe` probed | 0.822 | 0.43 ms | 0.20 s |
| `hnsw` | small-world graph, ef beam | 0.895 | 1.14 ms | 14.3 s |

Measured on 5k clustered vectors, dim 64, 100 queries (`bench.py`,
committed in `results/scan_benchmark.json`). The honest read at this
scale: **flat wins on every axis that matters** — exact and nearly as
fast. `ivf` buys 3.5x query speed for 18 points of recall. The
pure-Python HNSW is the cautionary tale: 70x slower to build than flat
and worse recall than IVF — `ef_construction` is the quality knob
(200 vs 64 moved recall 0.65 -> 0.90), and a production HNSW earns its
keep only via compiled index structures this implementation
deliberately lacks.

Two real bugs the recall benchmark caught, kept in `git log`:
layer-descent re-entered from the just-inserted node (self-distance 0)
and orphaned it; nearest-M pruning deleted long-range bridge edges and
fragmented the graph (only 104/800 nodes reachable). The
relative-neighborhood heuristic in `_prune` is the fix.

## Not implemented (the real boundary)

- Distributed/remote serving, replication, sharding
- Sparse/hybrid indexes, hosted reranker models, GPU acceleration

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
