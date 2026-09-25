"""vdb CLI + entrypoint.

  vdb serve                          MCP stdio server
  vdb create NAME --dim 384 [--metric cosine]
  vdb list | describe NAME | stats NAME | drop NAME
  vdb upsert NAME --ns NS --file records.jsonl   (id, values, metadata)
  vdb query NAME --vector-file v.json | --id ID [--ns NS] [--top-k 10] [--filter JSON]
  vdb fetch NAME --ids a b c [--ns NS]
  vdb delete NAME [--ns NS] [--ids ...] [--all]
"""

import argparse
import json
import sys
from pathlib import Path

from vdb_mcp.store import Store


def _records(path: str) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(prog="vdb")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve")
    c = sub.add_parser("create")
    c.add_argument("name")
    c.add_argument("--dim", type=int, required=True)
    c.add_argument("--metric", default="cosine")
    sub.add_parser("list")
    d = sub.add_parser("describe")
    d.add_argument("name")
    st = sub.add_parser("stats")
    st.add_argument("name")
    dr = sub.add_parser("drop")
    dr.add_argument("name")

    u = sub.add_parser("upsert")
    u.add_argument("name")
    u.add_argument("--ns", default="")
    u.add_argument("--file", required=True)

    q = sub.add_parser("query")
    q.add_argument("name")
    q.add_argument("--ns", default="")
    q.add_argument("--vector-file")
    q.add_argument("--id")
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--filter", default=None)

    f = sub.add_parser("fetch")
    f.add_argument("name")
    f.add_argument("--ns", default="")
    f.add_argument("--ids", nargs="+", required=True)

    dl = sub.add_parser("delete")
    dl.add_argument("name")
    dl.add_argument("--ns", default="")
    dl.add_argument("--ids", nargs="+")
    dl.add_argument("--all", action="store_true")

    args = ap.parse_args()
    if args.cmd == "serve":
        from vdb_mcp.server import main as serve_main
        serve_main()
        return

    s = Store()
    if args.cmd == "create":
        out = s.create_index(args.name, args.dim, args.metric)
    elif args.cmd == "list":
        out = s.list_indexes()
    elif args.cmd == "describe":
        out = s.describe_index(args.name)
    elif args.cmd == "stats":
        out = s.describe_index_stats(args.name)
    elif args.cmd == "drop":
        s.delete_index(args.name)
        out = {"deleted": args.name}
    elif args.cmd == "upsert":
        out = s.upsert(args.name, _records(args.file), args.ns)
    elif args.cmd == "query":
        vec = (json.loads(Path(args.vector_file).read_text())
               if args.vector_file else None)
        flt = json.loads(args.filter) if args.filter else None
        out = s.query(args.name, vector=vec, id=args.id,
                      top_k=args.top_k, namespace=args.ns, filter=flt)
    elif args.cmd == "fetch":
        out = s.fetch(args.name, args.ids, args.ns)
    elif args.cmd == "delete":
        out = s.delete(args.name, args.ns, args.ids, args.all)
    else:
        sys.exit(2)
    print(json.dumps(out, indent=1))
