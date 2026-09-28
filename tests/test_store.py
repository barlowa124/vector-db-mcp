import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vdb_mcp.index import VectorIndex
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


BAD_NAMES = ["../x", "a/b", "a\\b", "C:\\x", "D:rel", "\\\\host\\share",
             "a:b", ".hidden", "x.tmp", "foo.", "foo ", "..", "con",
             "nul.txt", "a|b", "a?b", "a*b", "a<b", 'a"b', "a\x00b", "", 123]


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_invalid_index_names_rejected(tmp_path, bad):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    s.upsert("docs", [{"id": "a", "values": [1, 0]}])
    with pytest.raises(ValueError):
        s.create_index(bad, 2)
    with pytest.raises(ValueError):
        s.get(bad)
    with pytest.raises(ValueError):
        s.delete_index(bad)
    assert [i["name"] for i in s.list_indexes()] == ["docs"]


BAD_NS = ["D:escape", "D:", "con", "nul.txt", "trail.", "trail ",
          "a|b", "a\x00b"]


@pytest.mark.parametrize("bad", BAD_NS)
def test_invalid_namespaces_rejected(tmp_path, bad):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    s.upsert("docs", [{"id": "original", "values": [1, 0]}])
    with pytest.raises(ValueError):
        s.upsert("docs", [{"id": "x", "values": [0, 1]}], bad)
    with pytest.raises(ValueError):
        s.query("docs", vector=[1, 0], namespace=bad)
    with pytest.raises(ValueError):
        s.fetch("docs", ["original"], bad)
    with pytest.raises(ValueError):
        s.delete("docs", bad, ids=["original"])
    with pytest.raises(ValueError):
        s.update("docs", bad, "original", values=[1, 1])
    fetched = Store(tmp_path).get("docs").fetch(["original"])["vectors"]
    assert fetched["original"]["values"] == pytest.approx([1, 0])
    assert [p.name for p in tmp_path.iterdir()] == ["docs"]
    assert {p.name for p in (tmp_path / "docs").iterdir()} == {
        "index.json", "_default"}


def test_symlink_index_rejected_without_touching_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("fixture")
    root = tmp_path / "indexes"
    root.mkdir()
    os.symlink(outside, root / "link")
    s = Store(root)
    for op in (lambda: s.create_index("link", 2),
               lambda: s.get("link"),
               lambda: s.delete_index("link")):
        with pytest.raises(ValueError):
            op()
    assert marker.read_text() == "fixture"
    assert (root / "link").is_symlink()


def test_dotted_name_roundtrip(tmp_path):
    s = Store(tmp_path)
    s.create_index("foo.bar", 2)
    s.upsert("foo.bar", [{"id": "a", "values": [1, 0]}])
    idx = Store(tmp_path).get("foo.bar")
    assert "a" in idx.fetch(["a"])["vectors"]


def _assert_originals_intact(tmp_path, s):
    for store in (Store(tmp_path), s):
        fetched = store.get("docs").fetch(["original", "new"])["vectors"]
        assert fetched["original"]["values"] == pytest.approx([1, 0])
        assert "new" not in fetched


def test_failed_final_swap_keeps_original(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    s.upsert("docs", [{"id": "original", "values": [1, 0]}])
    real_replace = Path.replace

    def flaky(self, target):
        if self.name.startswith(".") and ".stage-" in self.name:
            raise OSError("injected stage swap failure")
        return real_replace(self, target)

    with patch.object(Path, "replace", flaky):
        with pytest.raises(OSError):
            s.upsert("docs", [{"id": "original", "values": [0, 1]},
                              {"id": "new", "values": [0, 1]}])
    _assert_originals_intact(tmp_path, s)
    assert not [p for p in tmp_path.iterdir() if ".stage-" in p.name]


def test_failed_record_write_keeps_original(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    s.upsert("docs", [{"id": "original", "values": [1, 0]}])
    with patch("vdb_mcp.index.np.savez_compressed",
               side_effect=OSError("injected write failure")):
        with pytest.raises(OSError):
            s.upsert("docs", [{"id": "original", "values": [0, 1]},
                              {"id": "new", "values": [0, 1]}])
    _assert_originals_intact(tmp_path, s)


def test_delete_index_removes_retained_backup(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    s.upsert("docs", [{"id": "a", "values": [1, 0]}])
    shutil.copytree(tmp_path / "docs", tmp_path / ".docs.backup")
    s.delete_index("docs")
    with pytest.raises(KeyError):
        Store(tmp_path).get("docs")
    assert Store(tmp_path).list_indexes() == []
    assert not (tmp_path / "docs").exists()
    assert not (tmp_path / ".docs.backup").exists()


def test_delete_index_restores_backup_then_deletes(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    s.upsert("docs", [{"id": "a", "values": [1, 0]}])
    os.rename(tmp_path / "docs", tmp_path / ".docs.backup")
    s.delete_index("docs")
    with pytest.raises(KeyError):
        Store(tmp_path).get("docs")
    assert Store(tmp_path).list_indexes() == []
    assert not (tmp_path / ".docs.backup").exists()


def test_list_indexes_surfaces_recovery_error(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    os.rename(tmp_path / "docs", tmp_path / ".docs.backup")
    with patch.object(VectorIndex, "recover",
                      side_effect=OSError("disk gone")):
        with pytest.raises(OSError):
            Store(tmp_path).list_indexes()


def test_get_validates_path_before_cache(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    s.get("docs")
    os.rename(tmp_path / "docs", tmp_path / "real")
    os.symlink(tmp_path / "real", tmp_path / "docs")
    with pytest.raises(ValueError):
        s.get("docs")


def test_crash_backup_recovery(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    s.upsert("docs", [{"id": "a", "values": [1, 0]}])
    os.rename(tmp_path / "docs", tmp_path / ".docs.backup")
    assert "a" in Store(tmp_path).get("docs").fetch(["a"])["vectors"]
    assert [i["name"] for i in Store(tmp_path).list_indexes()] == ["docs"]


def test_hidden_stage_dirs_ignored(tmp_path):
    s = Store(tmp_path)
    s.create_index("docs", 2)
    leftover = tmp_path / ".docs.stage-xyz"
    leftover.mkdir()
    (leftover / "index.json").write_text("{}")
    assert [i["name"] for i in s.list_indexes()] == ["docs"]
    assert leftover.is_dir()


def test_concurrent_upserts_retain_all_ids(tmp_path):
    s = Store(tmp_path)
    s.create_index("m", 2)

    def worker(i):
        s.upsert("m", [{"id": f"id-{i}", "values": [float(i), 0.0]}])

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(40)))
    got = Store(tmp_path).get("m").fetch(
        [f"id-{i}" for i in range(40)])["vectors"]
    assert set(got) == {f"id-{i}" for i in range(40)}
