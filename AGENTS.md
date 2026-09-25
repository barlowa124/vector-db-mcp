# Project Guidance

- The MCP tool names and semantics mirror Pinecone's API surface. Do not
  rename them or drift the semantics (upsert overwrites by id, queries
  are namespace-scoped, euclidean score is negative distance).
- Search is exact by design — never describe results as approximate, and
  never add an ANN index without documenting the recall trade-off.
- Writes persist immediately via tmp-dir + rename; do not add an
  in-memory-only write path.
- Checkpoints/index data stay out of git (`data/` is gitignored).
- Run `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q` after changes.
