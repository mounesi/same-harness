#!/usr/bin/env python3
"""generate_partitions.py — freeze the train / dev / final_holdout split (CONTRACTS.md §6.2).

    ##############################################################################
    #  THIS FILE IS FROZEN BEFORE ANY BENCHMARK RUN AND NEVER REGENERATED AGAIN. #
    #                                                                            #
    #  Regenerating suites/partitions.json after Phase 1 has produced any run     #
    #  INVALIDATES THE STUDY: the Phase 2 LoRA leakage guard is the claim that    #
    #  no final-holdout task was ever visible to training, and that claim rests   #
    #  entirely on this file predating every run. A reshuffle silently moves      #
    #  tasks the model already trained on into the holdout, and no downstream     #
    #  check can detect it after the fact.                                        #
    #                                                                            #
    #  If the split genuinely has to change, it is a NEW FILE (partitions-v2.json)#
    #  and a NEW PROJECT PHASE with its own runs — never an edit to this one.     #
    #  training/build_dataset.py compiles final_holdout_sha256 in as a source     #
    #  constant and exits 3 if the file it loads disagrees.                       #
    ##############################################################################

Usage:

    python3 suites/generate_partitions.py                    # write suites/partitions.json
    python3 suites/generate_partitions.py --verify           # prove the committed file reproduces
    python3 suites/generate_partitions.py --print-final-holdout-sha256
                                                             # the constant for build_dataset.py

Method: seeded, stratified by suite, 60/20/20 within each suite. Ids are fully qualified
(`suite::instance_id`) so the three suites share one namespace. The union of the three
partitions must equal exactly the union of the three seed files — no id missing, none twice.

Ordering of the whole project: seed files are generated first, partitions second, runs third.
While the seed files are still placeholders this file is marked `"placeholder": true` and
must be regenerated (once) together with them.

Exit codes: 0 ok, 1 verification failed, 2 usage/config.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

SCHEMA = "partitions/v1"
DEFAULT_SEED = 20260830
FROZEN_BY = "AI-P153 phase-0"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "suites" / "partitions.json"

# Fixed order — the RNG stream depends on it, so it is part of the algorithm.
SEED_FILES: list[tuple[str, str]] = [
    ("swebench-verified", "suites/verified-100.json"),
    ("swebench-pro", "suites/pro-50.json"),
    ("agenttask", "suites/agenttask/seed.json"),
]

SPLIT = {"train": 60, "dev": 20}  # percent; final_holdout takes the remainder
METHOD = "seeded-stratified-by-suite  (60/20/20 within each suite)"
ALGORITHM = (
    "rng = random.Random(seed); "
    "for suite in ['swebench-verified','swebench-pro','agenttask']: "
    "s = rng.sample(sorted(qualified_ids[suite]), n); "
    "train += s[:n*60//100]; dev += s[n*60//100:n*60//100+n*20//100]; "
    "final_holdout += s[n*60//100+n*20//100:]  # CPython 3.11"
)
PLACEHOLDER_FROZEN_AT = "2026-08-30T00:00:00Z"
PLACEHOLDER_TODO = (
    "PLACEHOLDER — derived from placeholder seed files. Regenerate this file ONCE, together "
    "with the real seed files, BEFORE the first benchmark run, then never again. See the "
    "banner in suites/generate_partitions.py."
)


def die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def canonical_dumps(doc: Any) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_seed_files() -> tuple[dict[str, list[str]], list[dict], bool]:
    """Returns (suite -> qualified ids, per-file provenance records, any_placeholder)."""
    by_suite: dict[str, list[str]] = {}
    sources: list[dict] = []
    placeholder = False
    for suite, rel in SEED_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            die(f"seed file missing: {rel} — generate the seed files before the partitions")
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("suite") != suite:
            die(f"{rel}: suite is {doc.get('suite')!r}, expected {suite!r}")
        ids = list(doc.get("instance_ids") or [])
        if doc.get("count") != len(ids):
            die(f"{rel}: count != len(instance_ids)")
        if len(set(ids)) != len(ids):
            die(f"{rel}: duplicate instance_ids")
        placeholder = placeholder or bool(doc.get("placeholder"))
        by_suite[suite] = [f"{suite}::{i}" for i in ids]
        sources.append(
            {
                "suite": suite,
                "path": rel,
                "sha256": file_sha256(path),
                "count": len(ids),
                "instance_ids_sha256": doc.get("instance_ids_sha256"),
                "placeholder": bool(doc.get("placeholder", False)),
            }
        )
    return by_suite, sources, placeholder


def partition(by_suite: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out: dict[str, list[str]] = {"train": [], "dev": [], "final_holdout": []}
    for suite, _rel in SEED_FILES:
        ids = sorted(by_suite[suite])
        n = len(ids)
        shuffled = rng.sample(ids, n)
        n_train = n * SPLIT["train"] // 100
        n_dev = n * SPLIT["dev"] // 100
        out["train"] += shuffled[:n_train]
        out["dev"] += shuffled[n_train : n_train + n_dev]
        out["final_holdout"] += shuffled[n_train + n_dev :]
    return {k: sorted(v) for k, v in out.items()}


def validate(parts: dict[str, list[str]], by_suite: dict[str, list[str]]) -> list[str]:
    everything = [i for ids in by_suite.values() for i in ids]
    union: list[str] = []
    for name, ids in parts.items():
        if len(set(ids)) != len(ids):
            die(f"partition {name} contains duplicates")
        union += ids
    if len(set(union)) != len(union):
        die("partitions overlap — an id appears in more than one partition")
    if set(union) != set(everything):
        missing = sorted(set(everything) - set(union))
        extra = sorted(set(union) - set(everything))
        die(f"partition union != seed union (missing {len(missing)}, extra {len(extra)})")
    return sorted(union)


def build_doc(seed: int, frozen_at: str) -> dict:
    by_suite, sources, placeholder = load_seed_files()
    parts = partition(by_suite, seed)
    union = validate(parts, by_suite)

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "frozen_at": frozen_at,
        "frozen_by": FROZEN_BY,
        "seed": seed,
        "method": METHOD,
        "algorithm": ALGORITHM,
        "generator": "suites/generate_partitions.py",
        "source_seed_files": sources,
        "policy": {
            "train_usable_by": ["training/build_dataset.py"],
            "dev_usable_by": ["training/build_dataset.py --split dev", "hyperparameter selection"],
            "final_holdout_usable_by": ["analysis/aggregate.py reporting only"],
            "final_holdout_write_once": True,
            "final_holdout_must_never_enter_training": True,
        },
        "partitions": {
            name: {"count": len(ids), "ids": ids} for name, ids in sorted(parts.items())
        },
        "checksums": {
            "train_sha256": ids_sha256(parts["train"]),
            "dev_sha256": ids_sha256(parts["dev"]),
            "final_holdout_sha256": ids_sha256(parts["final_holdout"]),
            "all_sha256": ids_sha256(union),
        },
    }
    if placeholder:
        doc["placeholder"] = True
        doc["todo"] = PLACEHOLDER_TODO
    return doc


# --- commands ---------------------------------------------------------------------------


def cmd_write(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else OUT_PATH
    frozen_at = PLACEHOLDER_FROZEN_AT
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if not existing.get("placeholder") and not args.force:
            die(
                f"{out} is already FROZEN. Regenerating it after any run invalidates the study "
                "(see the banner in this file). If you are certain nothing has run yet, pass "
                "--force; otherwise create partitions-v2.json in a new project phase."
            )
    doc = build_doc(args.seed, frozen_at if args.deterministic_timestamp else now_utc())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_dumps(doc), encoding="utf-8")

    counts = {k: v["count"] for k, v in doc["partitions"].items()}
    print(f"==> wrote {out} {counts}", file=sys.stderr)
    if doc.get("placeholder"):
        print("==> WARNING: built from PLACEHOLDER seed files — regenerate before Phase 1", file=sys.stderr)
    print(f"==> final_holdout_sha256 = {doc['checksums']['final_holdout_sha256']}", file=sys.stderr)
    print(f"{out} {doc['checksums']['all_sha256']}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else OUT_PATH
    if not out.is_file():
        die(f"{out} not found")
    doc = json.loads(out.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        die(f"{out}: schema is {doc.get('schema')!r}, expected {SCHEMA!r}", 1)

    rebuilt = build_doc(doc.get("seed", DEFAULT_SEED), doc.get("frozen_at", ""))
    if canonical_dumps(rebuilt) != canonical_dumps(doc):
        die(
            f"{out} does NOT reproduce from its recorded seed and the current seed files. "
            "Either the split or a seed file changed after the freeze.",
            1,
        )
    print(f"{out} OK  final_holdout_sha256={doc['checksums']['final_holdout_sha256']}")
    return 0


def cmd_print_holdout(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else OUT_PATH
    if not out.is_file():
        die(f"{out} not found")
    doc = json.loads(out.read_text(encoding="utf-8"))
    print(doc["checksums"]["final_holdout_sha256"])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output path (default: suites/partitions.json)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"split seed (default {DEFAULT_SEED})")
    p.add_argument("--force", action="store_true", help="overwrite an already-frozen partitions file")
    p.add_argument(
        "--deterministic-timestamp",
        action="store_true",
        help="stamp the fixed placeholder timestamp instead of now (used for the committed placeholder)",
    )
    p.add_argument("--verify", action="store_true", help="check the committed file reproduces")
    p.add_argument(
        "--print-final-holdout-sha256",
        action="store_true",
        help="print the constant training/build_dataset.py compiles in",
    )
    args = p.parse_args(argv)

    if args.verify:
        return cmd_verify(args)
    if args.print_final_holdout_sha256:
        return cmd_print_holdout(args)
    return cmd_write(args)


if __name__ == "__main__":
    raise SystemExit(main())
