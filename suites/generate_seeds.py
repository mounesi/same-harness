#!/usr/bin/env python3
"""generate_seeds.py — deterministic, seeded subset selection for the benchmark suites.

Writes `suite-seed/v1` files (CONTRACTS.md §6.1): ids only, never task text. The point of
this script is that the subset is *reproducible* — anyone with the same population list and
the recorded seed regenerates the committed file byte for byte.

    # regenerate a suite from a population id list (see --population formats below)
    python3 suites/generate_seeds.py --suite swebench-verified \
        --population /tmp/verified-ids.txt --revision <dataset git sha>

    # CI / review: prove the committed file still matches its recorded selection
    python3 suites/generate_seeds.py --verify suites/verified-100.json \
        --population /tmp/verified-ids.txt

    # regenerate the committed PLACEHOLDERS (no dataset access needed)
    python3 suites/generate_seeds.py --placeholder --suite swebench-pro

Selection methods:
  seeded-uniform-without-replacement   random.Random(seed).sample(sorted(population), count)
  full-enumeration                     sorted(population)  — used by agenttask (all 50 tasks)

Population file formats (--population): a `.txt` of one id per line (`#` comments allowed),
a `.json` list of ids or of objects with an `instance_id` key, or a `.jsonl` of such objects.
This script never touches the network: fetching the dataset and dumping its ids is a separate,
deliberate step, so that what gets committed is auditable.

CONTRACTS.md §6.1 refers to this program as `suites/select.py`; that name exists as a thin
shim so the documented CI invocation works. This file is the implementation.

Exit codes: 0 ok, 1 verification failed, 2 usage/config.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "suite-seed/v1"
SELECTOR = "suites/generate_seeds.py"
SELECTOR_VERSION = "1.0.0"
DEFAULT_SEED = 20260830

METHOD_SAMPLE = "seeded-uniform-without-replacement"
METHOD_FULL = "full-enumeration"

INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
REPO_ROOT = Path(__file__).resolve().parents[1]

# Placeholder files must be byte-reproducible, so their timestamps are frozen constants.
PLACEHOLDER_FROZEN_AT = "2026-08-30T00:00:00Z"
PLACEHOLDER_REVISION = "TODO-UNRESOLVED-DATASET-REVISION"
PLACEHOLDER_TODO = (
    "PLACEHOLDER — NOT A REAL SELECTION. These ids are synthetic. Regenerate this file with "
    "suites/generate_seeds.py once dataset access is confirmed, then re-freeze "
    "suites/partitions.json. No benchmark run may use a placeholder seed file."
)

SUITES: dict[str, dict[str, Any]] = {
    "swebench-verified": {
        "out": "suites/verified-100.json",
        "count": 100,
        "method": METHOD_SAMPLE,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "tag": "verified",
    },
    "swebench-pro": {
        "out": "suites/pro-50.json",
        "count": 50,
        "method": METHOD_SAMPLE,
        # TODO verify the exact hub id for SWE-bench Pro when access is confirmed.
        "dataset": "ScaleAI/SWE-bench_Pro",
        "split": "test",
        "tag": "pro",
    },
    "agenttask": {
        "out": "suites/agenttask/seed.json",
        "count": 50,
        "method": METHOD_FULL,
        "dataset": "internal/agenttask",
        "split": "all",
        "tag": "agenttask",
    },
}


# --- helpers ----------------------------------------------------------------------------


def die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def canonical_dumps(doc: Any) -> str:
    """On-disk JSON per CONTRACTS.md §0: indent=2, sort_keys, one trailing newline."""
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode("utf-8")).hexdigest()


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_population(path: Path) -> list[str]:
    if not path.is_file():
        die(f"population file not found: {path}")
    text = path.read_text(encoding="utf-8")
    ids: list[str] = []
    if path.suffix == ".json":
        doc = json.loads(text)
        if not isinstance(doc, list):
            die(f"{path}: expected a JSON list")
        ids = [d if isinstance(d, str) else str(d["instance_id"]) for d in doc]
    elif path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                ids.append(str(json.loads(line)["instance_id"]))
    else:
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line)
    if not ids:
        die(f"{path}: no ids found")
    return ids


def validate_ids(ids: list[str], where: str) -> None:
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        die(f"{where}: duplicate ids: {dupes[:5]}")
    bad = [i for i in ids if not INSTANCE_ID_RE.match(i)]
    if bad:
        die(f"{where}: ids must match ^[A-Za-z0-9._-]+$ (they become directory names): {bad[:5]}")


def select(population: list[str], method: str, seed: int, count: int) -> tuple[list[str], str]:
    """Return (selected ids in selection order, the literal algorithm one-liner)."""
    pop = sorted(population)
    if method == METHOD_SAMPLE:
        if count > len(pop):
            die(f"cannot sample {count} ids from a population of {len(pop)}")
        chosen = random.Random(seed).sample(pop, count)
        algorithm = f"random.Random(seed).sample(sorted(population_ids), {count})  # CPython 3.11"
    elif method == METHOD_FULL:
        if count != len(pop):
            die(f"full-enumeration expects count == population size ({count} != {len(pop)})")
        chosen = pop
        algorithm = f"sorted(population_ids)  # full enumeration of all {count}, no sampling"
    else:
        die(f"unknown selection method: {method}")
    return chosen, algorithm


def synthetic_population(tag: str, count: int, method: str) -> list[str]:
    """The stand-in population behind the committed placeholder files. Obviously fake by
    construction, so no one can mistake a placeholder for a real selection."""
    size = count if method == METHOD_FULL else count * 5
    return [f"PLACEHOLDER-{tag}-{i:04d}" for i in range(1, size + 1)]


def build_doc(
    *,
    suite: str,
    population: list[str],
    seed: int,
    count: int,
    method: str,
    dataset: str,
    revision: str,
    split: str,
    frozen_at: str,
    selected_at: str,
    placeholder: bool,
) -> dict:
    validate_ids(population, "population")
    chosen, algorithm = select(population, method, seed, count)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "suite": suite,
        "frozen_at": frozen_at,
        "source": {
            "dataset": dataset,
            "revision": revision,
            "split": split,
            "population_size": len(population),
            "population_ids_sha256": ids_sha256(population),
        },
        "selection": {
            "method": method,
            "seed": seed,
            "algorithm": algorithm,
            "selector": SELECTOR,
            "selector_version": SELECTOR_VERSION,
            "selected_at": selected_at,
            "stratified_by": None,
        },
        "count": len(chosen),
        "instance_ids": chosen,
        "instance_ids_sha256": ids_sha256(chosen),
    }
    if placeholder:
        doc["placeholder"] = True
        doc["todo"] = PLACEHOLDER_TODO
    return doc


def check_structure(doc: dict, path: Path) -> None:
    if doc.get("schema") != SCHEMA:
        die(f"{path}: schema is {doc.get('schema')!r}, expected {SCHEMA!r}", 1)
    suite = doc.get("suite")
    if suite not in SUITES:
        die(f"{path}: unknown suite {suite!r}", 1)
    ids = doc.get("instance_ids") or []
    validate_ids(ids, str(path))
    if doc.get("count") != len(ids):
        die(f"{path}: count {doc.get('count')} != len(instance_ids) {len(ids)}", 1)
    if doc.get("instance_ids_sha256") != ids_sha256(ids):
        die(f"{path}: instance_ids_sha256 does not match instance_ids", 1)
    if doc["selection"]["method"] not in (METHOD_SAMPLE, METHOD_FULL):
        die(f"{path}: unknown selection method", 1)


# --- commands ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    spec = SUITES[args.suite]
    count = args.count or spec["count"]
    method = args.method or spec["method"]
    out = Path(args.out) if args.out else REPO_ROOT / spec["out"]

    if args.placeholder:
        population = synthetic_population(spec["tag"], count, method)
        revision = PLACEHOLDER_REVISION
        frozen_at = selected_at = PLACEHOLDER_FROZEN_AT
    else:
        if not args.population:
            die("--population is required (or use --placeholder)")
        population = load_population(Path(args.population))
        revision = args.revision or ""
        if not revision:
            die("--revision is required: the seed file must pin the dataset revision it sampled")
        frozen_at = selected_at = now_utc()

    doc = build_doc(
        suite=args.suite,
        population=population,
        seed=args.seed,
        count=count,
        method=method,
        dataset=args.dataset or spec["dataset"],
        revision=revision,
        split=args.split or spec["split"],
        frozen_at=frozen_at,
        selected_at=selected_at,
        placeholder=bool(args.placeholder),
    )

    if out.exists() and not args.force:
        existing = json.loads(out.read_text(encoding="utf-8"))
        if not existing.get("placeholder"):
            die(f"{out} already exists and is not a placeholder — pass --force if you really mean it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_dumps(doc), encoding="utf-8")
    print(f"==> wrote {out} ({doc['count']} ids, {doc['instance_ids_sha256'][:12]}…)", file=sys.stderr)
    print(f"{out} {doc['count']} {doc['instance_ids_sha256']}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.verify)
    if not path.is_file():
        die(f"{path} not found")
    doc = json.loads(path.read_text(encoding="utf-8"))
    check_structure(doc, path)

    if doc.get("placeholder") and not args.allow_placeholder:
        die(
            f"{path} is a PLACEHOLDER seed file — regenerate it from the real dataset before "
            "any benchmark run (pass --allow-placeholder to check its self-consistency only)",
            1,
        )

    spec = SUITES[doc["suite"]]
    if doc.get("placeholder"):
        population = synthetic_population(spec["tag"], doc["count"], doc["selection"]["method"])
    elif args.population:
        population = load_population(Path(args.population))
    else:
        print(f"{path} OK (structure only — pass --population to re-run the selection)")
        return 0

    rebuilt = build_doc(
        suite=doc["suite"],
        population=population,
        seed=doc["selection"]["seed"],
        count=doc["count"],
        method=doc["selection"]["method"],
        dataset=doc["source"]["dataset"],
        revision=doc["source"]["revision"],
        split=doc["source"]["split"],
        frozen_at=doc["frozen_at"],
        selected_at=doc["selection"]["selected_at"],
        placeholder=bool(doc.get("placeholder")),
    )
    if canonical_dumps(rebuilt) != canonical_dumps(doc):
        die(f"{path} does NOT reproduce from its recorded seed/algorithm/population", 1)
    print(f"{path} OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=sorted(SUITES), help="suite to generate")
    p.add_argument("--population", help="file of candidate instance ids (.txt/.json/.jsonl)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"selection seed (default {DEFAULT_SEED})")
    p.add_argument("--count", type=int, help="how many to select (default: the suite's size)")
    p.add_argument("--method", choices=[METHOD_SAMPLE, METHOD_FULL], help="override the selection method")
    p.add_argument("--dataset", help="source dataset id recorded in the file")
    p.add_argument("--revision", help="source dataset revision — REQUIRED for a real selection")
    p.add_argument("--split", help="source split recorded in the file")
    p.add_argument("--out", help="output path (default: the suite's committed path)")
    p.add_argument("--placeholder", action="store_true", help="regenerate the synthetic placeholder file")
    p.add_argument("--force", action="store_true", help="overwrite a non-placeholder file")
    p.add_argument("--verify", metavar="SEED_FILE", help="check a committed seed file instead of writing one")
    p.add_argument("--allow-placeholder", action="store_true", help="--verify: tolerate a placeholder file")
    args = p.parse_args(argv)

    if args.verify:
        return cmd_verify(args)
    if not args.suite:
        p.error("--suite is required (or use --verify)")
    return cmd_generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
