"""Build a tiny synthetic AgentTask data pack, so the harness has something real to run.

Two tasks, each a three-file python package with a genuine failing test:
    smoke-0001  divide() raises ZeroDivisionError; the hidden test wants None
    smoke-0002  same shape, different module, so the pack is not a single-task special case

The pack is written OUTSIDE the repo by default ($TMPDIR) because a data pack carries task
content, and CONTRACTS §7.4 keeps that out of git. Nothing here touches the real
customer-derived suite — this is synthetic, publication-safe, and exists only to prove the
plumbing works end to end without a GPU.

Usage: python3 smoke/make_pack.py --out /tmp/smoke-pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tarfile
import tempfile

SCHEMA_TASK = "agenttask-task/v1"
SCHEMA_PACK = "agenttask-pack/v1"

BROKEN = '''"""Arithmetic helpers."""


def divide(a, b):
    return a / b


def add(a, b):
    return a + b
'''

VISIBLE_TEST = '''from {mod}.ops import add


def test_add():
    assert add(2, 3) == 5
'''

HIDDEN_TEST = '''from {mod}.ops import divide


def test_divide_by_zero_returns_none():
    assert divide(1, 0) is None


def test_divide_normal():
    assert divide(6, 3) == 2
'''

PROBLEM = """divide() crashes on a zero denominator.

Calling `divide(1, 0)` raises ZeroDivisionError. Callers in the reporting path cannot
distinguish that from a genuine error, so division by zero should return None instead of
raising. Fix `{mod}/ops.py` so that `divide(1, 0)` returns None while ordinary division is
unchanged.
"""


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_tree(tar: tarfile.TarFile, root: pathlib.Path, base: str) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            tar.add(str(path), arcname=os.path.join(base, str(path.relative_to(root))))


def build_task(out: pathlib.Path, iid: str, mod: str) -> dict:
    work = pathlib.Path(tempfile.mkdtemp(prefix="smoke-src-"))
    pkg = work / mod
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ops.py").write_text(BROKEN, encoding="utf-8")
    # A root conftest.py makes pytest put the workspace root on sys.path, so `from calc
    # import ...` resolves. Without it pytest only inserts tests/ and every test errors on
    # import (returncode 4) — which the grader would honestly report as 0/N passing.
    (work / "conftest.py").write_text("", encoding="utf-8")
    tests = work / "tests"
    tests.mkdir()
    (tests / "test_basic.py").write_text(VISIBLE_TEST.format(mod=mod), encoding="utf-8")

    snap = out / "snapshots" / ("%s.tar.gz" % iid)
    snap.parent.mkdir(parents=True, exist_ok=True)
    # deterministic tar: sorted entries, fixed mtime, no gzip timestamp
    with tarfile.open(str(snap), "w:gz", compresslevel=6) as tar:
        _add_tree(tar, work, ".")

    hidden_root = pathlib.Path(tempfile.mkdtemp(prefix="smoke-hid-"))
    htests = hidden_root / "tests"
    htests.mkdir(parents=True)
    (htests / "test_hidden.py").write_text(HIDDEN_TEST.format(mod=mod), encoding="utf-8")
    hid = out / "hidden-tests" / ("%s.tar.gz" % iid)
    hid.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(hid), "w:gz", compresslevel=6) as tar:
        _add_tree(tar, hidden_root, ".")

    record = {
        "schema": SCHEMA_TASK,
        "instance_id": iid,
        "repo": "internal/%s" % mod,
        "base_commit": "",
        "problem_statement": PROBLEM.format(mod=mod),
        "fail_to_pass": ["tests/test_hidden.py::test_divide_by_zero_returns_none"],
        "pass_to_pass": [
            "tests/test_hidden.py::test_divide_normal",
            "tests/test_basic.py::test_add",
        ],
        "snapshot": {"path": "snapshots/%s.tar.gz" % iid, "sha256": sha256_file(snap)},
        "hidden_tests": {"path": "hidden-tests/%s.tar.gz" % iid, "sha256": sha256_file(hid)},
        "environment": {
            "image": "",
            "setup_cmds": [],
            # No override: the adapter's DEFAULT_TEST_CMD uses `-rA`, whose short-summary
            # lines are what _parse_pytest_output reads. `-q` suppresses them and every
            # test parses as "not passed".
            "test_cmd": None,
            "runner": "pytest",
            "report_json": None,
            "test_timeout_s": 120,
            "setup_timeout_s": 120,
        },
        "labels": ["synthetic", "smoke"],
        "origin": "synthetic/smoke",
    }
    tasks_dir = out / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / ("%s.json" % iid)).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "smoke-pack"))
    args = ap.parse_args()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ids = ["smoke-0001", "smoke-0002"]
    mods = {"smoke-0001": "calc", "smoke-0002": "calc"}
    for iid in ids:
        build_task(out, iid, mods[iid])

    (out / "PACK.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA_PACK,
                "pack_revision": "smoke-1",
                "pack_sha256": None,
                "consent_class": "synthetic",
                "note": "Synthetic smoke pack — no customer content. Safe to publish.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    seed = {
        "schema": "suite-seed/v1",
        "suite": "agenttask",
        "frozen_at": "2026-08-31T00:00:00Z",
        "source": {"dataset": "synthetic/smoke", "revision": "smoke-1", "split": "test",
                   "population_size": len(ids), "population_ids_sha256": None},
        "selection": {"method": "full-enumeration", "seed": 0,
                      "algorithm": "all ids", "selector": "smoke/make_pack.py",
                      "selector_version": "1.0.0", "selected_at": "2026-08-31T00:00:00Z",
                      "stratified_by": None},
        "count": len(ids),
        "instance_ids": ids,
        # The seal manifest.py requires of every non-placeholder seed (§6.1): sha256 of the
        # sorted ids joined by newlines, trailing newline — the same recipe as
        # harness.manifest.id_set_sha256, so the freeze check passes for a genuine set.
        "instance_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(ids)) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    (out / "seed.json").write_text(json.dumps(seed, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    # Canonical partitions/v1 shape (CONTRACTS §6.2): nested under "partitions", each
    # block {count, ids}, ids fully qualified. smoke-0002 is deliberately final_holdout so
    # the smoke run also proves the Phase-2 leakage guard rejects it.
    parts = {
        "schema": "partitions/v1",
        "frozen_at": "2026-08-31T00:00:00Z",
        "frozen_by": "smoke/make_pack.py",
        "seed": 0,
        "method": "fixed (synthetic smoke pack)",
        "policy": {
            "final_holdout_write_once": True,
            "final_holdout_must_never_enter_training": True,
        },
        "partitions": {
            "train": {"count": 1, "ids": ["agenttask::smoke-0001"]},
            "dev": {"count": 0, "ids": []},
            "final_holdout": {"count": 1, "ids": ["agenttask::smoke-0002"]},
        },
    }
    (out / "partitions.json").write_text(json.dumps(parts, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
