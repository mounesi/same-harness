#!/usr/bin/env python3
"""build_dataset.py — turn Phase-1 run trajectories into LoRA training examples.

Phase 2 of AgentTask AI-P153 ("The Harness Variable") fine-tunes qwen3-coder-next on the
trajectories Phase 1 produced. That is only sound if the final holdout never touches training,
so the leakage guard here is code, not a comment:

  * the frozen ``final_holdout`` checksum is compiled into this file as FINAL_HOLDOUT_SHA256;
  * ``suites/partitions.json`` is re-hashed on every run and compared against it;
  * every candidate example's qualified id is asserted not to be a holdout id;
  * every input run manifest must agree on the partitions checksum it was run against;
  * the input is an EXPLICIT list of run manifests. Directory arguments are refused outright —
    this tool never discovers results by scanning.

Usage
-----
  build_dataset.py --manifests RUN1/run-manifest.json RUN2/run-manifest.json --out DIR
  build_dataset.py --manifest-list manifests.txt --split dev --out DIR
  build_dataset.py --freeze suites/partitions.json [--write]   # one-time, at partition freeze
  build_dataset.py --print-holdout-constant                    # what train_lora.sh cross-checks
  build_dataset.py --self-test                                 # unit-test the leakage guard

Exit codes:  0 ok · 2 usage/config · 3 leakage guard tripped (see docs/CONTRACTS.md §6.2)

Python 3.11 stdlib only. See docs/CONTRACTS.md §2 (manifests), §3 (raw results), §6.2 (partitions).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

BUILDER_VERSION = "1.0.0"
DATASET_SCHEMA = "lora-dataset/v1"
EXAMPLE_SCHEMA = "lora-example/v1"

# ---------------------------------------------------------------------------
# THE LEAKAGE GUARD CONSTANT.
#
# Set this ONCE, when suites/partitions.json is frozen, to the value printed by
#     python3 training/build_dataset.py --freeze suites/partitions.json
# (``--write`` patches the line below in place). While it holds the sentinel
# "UNFROZEN" this tool refuses to build anything: an unpinned holdout is
# indistinguishable from a tampered one.
# ---------------------------------------------------------------------------
FINAL_HOLDOUT_SHA256 = "UNFROZEN"

ALLOWED_SPLITS = ("train", "dev")
FORBIDDEN_SPLIT = "final_holdout"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_GUARD = 3


class ConfigError(Exception):
    """Bad inputs: missing files, malformed JSON, contradictory flags. Exit 2."""


class LeakageGuardError(Exception):
    """The train/dev/final-holdout boundary was violated or cannot be proven. Exit 3."""


# --------------------------------------------------------------- primitives --


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def id_list_sha256(ids: Sequence[str]) -> str:
    """CONTRACTS §6.2: sha256("\\n".join(sorted(ids)) + "\\n") — order independent."""
    return sha256_hex(("\n".join(sorted(ids)) + "\n").encode("utf-8"))


def qualified_id(suite: str, instance_id: str) -> str:
    return f"{suite}::{instance_id}"


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def constant_is_frozen(constant: str = FINAL_HOLDOUT_SHA256) -> bool:
    """True once FINAL_HOLDOUT_SHA256 has been pinned to a real digest."""
    return bool(HEX64.match(constant or ""))


# --------------------------------------------------------------- partitions --


def load_partitions(path: Path) -> dict[str, Any]:
    """Load and structurally validate suites/partitions.json. Raises ConfigError."""
    if not path.exists():
        raise ConfigError(
            f"partitions file not found: {path}\n"
            "The partition freeze (CONTRACTS §6.2) is a precondition for any training data."
        )
    if path.is_dir():
        raise ConfigError(f"--partitions must be a file, not a directory: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if doc.get("schema") != "partitions/v1":
        raise ConfigError(f"{path}: unexpected schema {doc.get('schema')!r}, want 'partitions/v1'")
    parts = doc.get("partitions")
    if not isinstance(parts, dict):
        raise ConfigError(f"{path}: missing 'partitions' object")
    for name in (*ALLOWED_SPLITS, FORBIDDEN_SPLIT):
        block = parts.get(name)
        if not isinstance(block, dict) or not isinstance(block.get("ids"), list):
            raise ConfigError(f"{path}: partitions.{name}.ids is missing or not a list")
        ids = block["ids"]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"{path}: partitions.{name}.ids contains duplicates")
        if block.get("count") is not None and block["count"] != len(ids):
            raise ConfigError(
                f"{path}: partitions.{name}.count={block['count']} != len(ids)={len(ids)}"
            )
    if doc.get("placeholder") is True:
        raise ConfigError(
            f"{path} is still marked \"placeholder\": true — it was generated from placeholder "
            "seed files and is not the frozen partition set. Regenerate it with the real seeds "
            "before building any training data (CONTRACTS §6.2)."
        )
    seen: dict[str, str] = {}
    for name in (*ALLOWED_SPLITS, FORBIDDEN_SPLIT):
        for qid in parts[name]["ids"]:
            if qid in seen:
                raise ConfigError(
                    f"{path}: id {qid!r} appears in both '{seen[qid]}' and '{name}' — "
                    "partitions must be disjoint"
                )
            seen[qid] = name
    return doc


def holdout_ids(partitions: dict[str, Any]) -> frozenset[str]:
    return frozenset(partitions["partitions"][FORBIDDEN_SPLIT]["ids"])


def split_ids(partitions: dict[str, Any], split: str) -> frozenset[str]:
    if split not in ALLOWED_SPLITS:
        raise ConfigError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
    return frozenset(partitions["partitions"][split]["ids"])


def partition_index(partitions: dict[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for name, block in partitions["partitions"].items():
        for qid in block["ids"]:
            index[qid] = name
    return index


def assert_partitions_frozen(
    partitions: dict[str, Any], expected: str = FINAL_HOLDOUT_SHA256
) -> str:
    """CONTRACTS §6.2 guard 1+2.

    Recompute the final-holdout checksum from the loaded file and require it to equal both the
    checksum the file records for itself and the digest compiled into this source. Returns the
    verified digest; raises LeakageGuardError otherwise.
    """
    if not constant_is_frozen(expected):
        raise LeakageGuardError(
            "FINAL_HOLDOUT_SHA256 is not pinned in training/build_dataset.py.\n"
            "Freeze it once, at partition-freeze time:\n"
            "    python3 training/build_dataset.py --freeze suites/partitions.json --write\n"
            "Refusing to build a dataset whose holdout boundary cannot be proven."
        )
    computed = id_list_sha256(sorted(holdout_ids(partitions)))
    recorded = (partitions.get("checksums") or {}).get("final_holdout_sha256")
    if recorded is not None and recorded != computed:
        raise LeakageGuardError(
            "partitions.json final_holdout has been modified since freeze: "
            f"ids hash to {computed}, but the file records {recorded}"
        )
    if computed != expected:
        raise LeakageGuardError(
            "partitions.json final_holdout has been modified since freeze: "
            f"ids hash to {computed}, but build_dataset.py was frozen against {expected}"
        )
    return computed


def assert_no_holdout(candidate_ids: Iterable[str], holdout: frozenset[str]) -> None:
    """CONTRACTS §6.2 guard 3. Never filters silently — any hit is fatal and named."""
    offenders = sorted({qid for qid in candidate_ids if qid in holdout})
    if offenders:
        shown = ", ".join(offenders[:10])
        more = f" (+{len(offenders) - 10} more)" if len(offenders) > 10 else ""
        raise LeakageGuardError(
            f"final-holdout task ids reached the training set: {shown}{more}. "
            "This is a leakage bug in the caller, not something to filter away."
        )


def assert_manifest_partitions(manifest: dict[str, Any], partitions_sha256: str, label: str) -> None:
    """CONTRACTS §6.2 guard 4: a run must have been executed against THIS partitions.json."""
    recorded = (manifest.get("suite") or {}).get("partitions_sha256")
    if not recorded:
        raise LeakageGuardError(
            f"{label}: manifest has no suite.partitions_sha256 — cannot prove which partition "
            "freeze this run was executed against"
        )
    if recorded != partitions_sha256:
        raise LeakageGuardError(
            f"{label}: run was executed against partitions_sha256={recorded}, but this build was "
            f"given a partitions.json hashing to {partitions_sha256}. Refusing to mix freezes."
        )


# ------------------------------------------------------------- git hygiene --


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_toplevel(path: Path):
    """The git work tree containing ``path`` (or its nearest existing ancestor), else None."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        proc = subprocess.run(
            ["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    top = proc.stdout.strip()
    return Path(top) if top else None


def assert_writable_outside_git(path: Path) -> None:
    """Datasets carry trajectory text (possibly restricted-consent). Never into git.

    Writing is allowed outside any git work tree, or inside one only if git already ignores the
    path (CONTRACTS §7.4 — results/, dist/, **/trajectories/ and friends must be gitignored).
    """
    path = path.resolve()
    top = git_toplevel(path)
    if top is None:
        return
    ignored = subprocess.run(
        ["git", "-C", str(top), "check-ignore", "-q", str(path)],
        capture_output=True,
        check=False,
    ).returncode == 0
    if ignored:
        return
    raise ConfigError(
        f"refusing to write the dataset inside the git work tree {top}: {path}\n"
        "Trajectory-derived data must never enter git (CONTRACTS §7.4). Write outside the "
        "repository, or add the path to .gitignore first."
    )


# ------------------------------------------------------------- run loading --


@dataclasses.dataclass(frozen=True)
class SourceRun:
    manifest_path: Path
    run_dir: Path
    manifest: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def suite(self) -> str:
        return str(self.manifest["suite"]["name"])

    @property
    def model(self) -> str:
        return str(self.manifest["model"]["name"])

    @property
    def consent_class(self) -> str:
        return str((self.manifest.get("flags") or {}).get("consent_class") or "restricted")


def read_manifest_arg(arg: str) -> Path:
    """Explicit-manifest rule (CONTRACTS §6.2 guard 4): no directories, ever."""
    path = Path(arg).expanduser()
    if path.is_dir():
        raise ConfigError(
            f"{arg} is a directory. build_dataset.py consumes an explicit list of run manifests; "
            "it never scans a results directory. Pass <run_dir>/run-manifest.json paths."
        )
    if not path.exists():
        raise ConfigError(f"manifest not found: {arg}")
    return path.resolve()


def read_manifest_list(path: Path) -> list[Path]:
    if path.is_dir():
        raise ConfigError(f"--manifest-list must be a file of paths, not a directory: {path}")
    if not path.exists():
        raise ConfigError(f"--manifest-list not found: {path}")
    out: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(read_manifest_arg(line))
    if not out:
        raise ConfigError(f"--manifest-list {path} contains no manifest paths")
    return out


def resolve_run_dir(manifest_path: Path, run_id: str, roots: Sequence[Path]) -> Path:
    """Locate the run directory holding results.jsonl for an explicitly named manifest.

    Candidates, in order: the manifest's own directory (the normal case — a run dir or an
    unpacked bundle), then <root>/<run_id> for each --runs-root. Only the exact run_id is ever
    looked up; no directory is enumerated.
    """
    candidates = [manifest_path.parent] + [root / run_id for root in roots]
    for cand in candidates:
        if (cand / "results.jsonl").is_file():
            return cand
    raise ConfigError(
        f"no results.jsonl found for run {run_id}. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + "\nFetch the bundle first:  ./resultsctl fetch "
        + run_id
        + " <dir>   then pass --runs-root <dir>"
    )


def load_source_runs(manifest_paths: Sequence[Path], roots: Sequence[Path]) -> list[SourceRun]:
    runs: list[SourceRun] = []
    seen: set[str] = set()
    for mp in manifest_paths:
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{mp} is not valid JSON: {exc}") from exc
        if manifest.get("schema") != "run-manifest/v1":
            raise ConfigError(f"{mp}: unexpected schema {manifest.get('schema')!r}")
        run_id = manifest.get("run_id")
        if not run_id:
            raise ConfigError(f"{mp}: manifest has no run_id")
        if run_id in seen:
            raise ConfigError(f"run {run_id} was given twice")
        seen.add(run_id)
        runs.append(SourceRun(mp, resolve_run_dir(mp, run_id, roots), manifest))
    if not runs:
        raise ConfigError("no run manifests given")
    return runs


def iter_results(run_dir: Path) -> Iterator[dict[str, Any]]:
    with (run_dir / "results.jsonl").open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{run_dir/'results.jsonl'}:{lineno}: {exc}") from exc


# ------------------------------------------------------- trajectory → chat --


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def trajectory_messages(records: Sequence[dict[str, Any]], max_tool_chars: int) -> list[dict[str, str]]:
    """Fold a trajectory JSONL (CONTRACTS §3.2) into an OpenAI-style message list.

    Tool results are truncated to ``max_tool_chars`` so one pathological ``cat`` cannot blow the
    sequence budget; the truncation is marked inline so it is visible in the data.
    """
    messages: list[dict[str, str]] = []
    for rec in sorted(records, key=lambda r: r.get("i", 0)):
        kind = rec.get("kind")
        role = rec.get("role")
        if kind == "error":
            continue
        content = _as_text(rec.get("content"))
        if kind == "tool_call":
            tool = rec.get("tool") or "tool"
            args = _as_text(rec.get("args", content))
            content = f"<tool_call name=\"{tool}\">{args}</tool_call>"
            role = "assistant"
        elif kind == "tool_result":
            role = "tool"
            if max_tool_chars > 0 and len(content) > max_tool_chars:
                dropped = len(content) - max_tool_chars
                content = content[:max_tool_chars] + f"\n… [{dropped} chars truncated]"
        if role not in ("system", "user", "assistant", "tool"):
            continue
        if not content:
            continue
        if messages and messages[-1]["role"] == role and kind == "completion":
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})
    return messages


def read_trajectory(run_dir: Path, record: dict[str, Any], verify_sha: bool) -> list[dict[str, Any]]:
    traj = record.get("trajectory") or {}
    ref = traj.get("ref")
    if not ref:
        raise ConfigError("result record has no trajectory.ref")
    if Path(ref).is_absolute() or ".." in str(ref).split("/"):
        raise ConfigError(f"trajectory.ref must be run-dir-relative, got {ref!r}")
    path = run_dir / ref
    if not path.is_file():
        raise ConfigError(f"missing trajectory file {path}")
    if verify_sha and traj.get("sha256"):
        got = sha256_path(path)
        if got != traj["sha256"]:
            raise ConfigError(
                f"{path}: sha256 {got} != recorded {traj['sha256']} — bundle is corrupt or edited"
            )
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------------ build ---


@dataclasses.dataclass
class BuildStats:
    considered: int = 0
    kept: int = 0
    skipped_wrong_split: int = 0
    skipped_unresolved: int = 0
    skipped_no_trajectory: int = 0
    skipped_too_few_messages: int = 0
    skipped_no_assistant: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def build_examples(
    runs: Sequence[SourceRun],
    *,
    wanted_ids: frozenset[str],
    holdout: frozenset[str],
    part_index: dict[str, str],
    split: str,
    require_resolved: bool,
    max_tool_chars: int,
    min_messages: int,
    verify_sha: bool,
    max_examples: int | None,
) -> tuple[list[dict[str, Any]], BuildStats]:
    examples: list[dict[str, Any]] = []
    stats = BuildStats()
    for run in runs:
        for rec in iter_results(run.run_dir):
            stats.considered += 1
            qid = qualified_id(run.suite, str(rec["instance_id"]))
            # Guard 3, applied to every single candidate before anything else happens with it.
            assert_no_holdout([qid], holdout)
            if qid not in wanted_ids:
                stats.skipped_wrong_split += 1
                continue
            if require_resolved and rec.get("resolved") is not True:
                stats.skipped_unresolved += 1
                continue
            traj = rec.get("trajectory") or {}
            if not traj.get("ref"):
                stats.skipped_no_trajectory += 1
                continue
            records = read_trajectory(run.run_dir, rec, verify_sha)
            messages = trajectory_messages(records, max_tool_chars)
            if len(messages) < min_messages:
                stats.skipped_too_few_messages += 1
                continue
            if not any(m["role"] == "assistant" for m in messages):
                stats.skipped_no_assistant += 1
                continue
            example_id = sha256_hex(
                f"{run.run_id}|{rec['instance_id']}|{rec.get('pass_idx', 0)}".encode("utf-8")
            )[:16]
            examples.append(
                {
                    "schema": EXAMPLE_SCHEMA,
                    "example_id": example_id,
                    "run_id": run.run_id,
                    "suite": run.suite,
                    "model": run.model,
                    "instance_id": rec["instance_id"],
                    "qualified_id": qid,
                    "partition": part_index.get(qid, "unpartitioned"),
                    "split": split,
                    "pass_idx": rec.get("pass_idx", 0),
                    "resolved": bool(rec.get("resolved")),
                    "error_code": rec.get("error_code"),
                    "consent_class": traj.get("consent_class", run.consent_class),
                    "messages": messages,
                    "meta": {
                        "iterations": rec.get("iterations"),
                        "tool_calls": rec.get("tool_calls"),
                        "tokens_total": (rec.get("tokens") or {}).get("total"),
                        "patch_ref": (rec.get("patch") or {}).get("ref"),
                        "trajectory_sha256": traj.get("sha256"),
                    },
                }
            )
            stats.kept += 1
            if max_examples is not None and len(examples) >= max_examples:
                return examples, stats
    return examples, stats


def write_dataset(
    out_dir: Path,
    name: str,
    examples: Sequence[dict[str, Any]],
    *,
    runs: Sequence[SourceRun],
    partitions_path: Path,
    partitions_sha256: str,
    holdout_sha256: str,
    split: str,
    stats: BuildStats,
    filters: dict[str, Any],
    force: bool = False,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = out_dir / "dataset-manifest.json"
    if existing.exists() and not force:
        try:
            prev = json.loads(existing.read_text(encoding="utf-8")).get("dataset_name")
        except json.JSONDecodeError:
            prev = None
        if prev is not None and prev != name:
            raise ConfigError(
                f"{existing} already describes dataset {prev!r}. Use a separate --out directory "
                f"per dataset (one manifest per dataset), or pass --force to overwrite."
            )
    data_path = out_dir / f"{name}.jsonl"
    with data_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, sort_keys=True, separators=(",", ":")) + "\n")
    consent = "restricted" if any(r.consent_class == "restricted" for r in runs) else "public"
    manifest = {
        "schema": DATASET_SCHEMA,
        "dataset_name": name,
        "builder_version": BUILDER_VERSION,
        "created_at": utcnow(),
        "split": split,
        "dataset_file": data_path.name,
        "dataset_sha256": sha256_path(data_path),
        "dataset_bytes": data_path.stat().st_size,
        "example_count": len(examples),
        "example_schema": EXAMPLE_SCHEMA,
        "consent_class": consent,
        "source_run_ids": sorted(r.run_id for r in runs),
        "source_manifests": [
            {
                "run_id": r.run_id,
                "path": str(r.manifest_path),
                "sha256": sha256_path(r.manifest_path),
                "suite": r.suite,
                "model": r.model,
                "status": r.manifest.get("status"),
                "harness_version": (r.manifest.get("harness") or {}).get("version"),
                "weight_digest": (r.manifest.get("model") or {}).get("weight_digest"),
            }
            for r in runs
        ],
        "partitions_file": str(partitions_path),
        "partitions_sha256": partitions_sha256,
        "final_holdout_sha256": holdout_sha256,
        "filters": filters,
        "stats": stats.as_dict(),
        "publication_rule": (
            "The tuned model is reported on the untouched final_holdout partition only. "
            "No example in this dataset comes from final_holdout — see CONTRACTS §6.2."
        ),
    }
    manifest_path = out_dir / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data_path, manifest_path


# ------------------------------------------------------------------ freeze --


def cmd_freeze(partitions_path: Path, write: bool) -> int:
    partitions = load_partitions(partitions_path)
    computed = id_list_sha256(sorted(holdout_ids(partitions)))
    recorded = (partitions.get("checksums") or {}).get("final_holdout_sha256")
    if recorded is not None and recorded != computed:
        print(
            f"error: {partitions_path} records final_holdout_sha256={recorded} but its ids hash "
            f"to {computed} — fix the file before freezing",
            file=sys.stderr,
        )
        return EXIT_GUARD
    print(f'FINAL_HOLDOUT_SHA256 = "{computed}"')
    if not write:
        print(
            "note: paste that line into training/build_dataset.py, or re-run with --write",
            file=sys.stderr,
        )
        return EXIT_OK
    src_path = Path(__file__).resolve()
    src = src_path.read_text(encoding="utf-8")
    pattern = re.compile(r'^FINAL_HOLDOUT_SHA256 = "[^"]*"$', re.M)
    if not pattern.search(src):
        print("error: could not find the FINAL_HOLDOUT_SHA256 line to patch", file=sys.stderr)
        return EXIT_CONFIG
    if constant_is_frozen() and FINAL_HOLDOUT_SHA256 != computed:
        print(
            "error: FINAL_HOLDOUT_SHA256 is already frozen to "
            f"{FINAL_HOLDOUT_SHA256} and would change to {computed}.\n"
            "       Partitions are frozen once (CONTRACTS §6.2); a change needs a new file name "
            "and a new project phase.",
            file=sys.stderr,
        )
        return EXIT_GUARD
    src_path.write_text(
        pattern.sub(f'FINAL_HOLDOUT_SHA256 = "{computed}"', src, count=1), encoding="utf-8"
    )
    print(f"==> pinned FINAL_HOLDOUT_SHA256 in {src_path}", file=sys.stderr)
    return EXIT_OK


# --------------------------------------------------------------- self-test --


class GuardTests(unittest.TestCase):
    HOLDOUT = ["swebench-pro::pandas__pandas-51284", "agenttask::at-0007"]
    TRAIN = ["swebench-verified::astropy__astropy-12907"]
    DEV = ["swebench-verified::django__django-11099"]

    def partitions(self) -> dict[str, Any]:
        return {
            "schema": "partitions/v1",
            "partitions": {
                "train": {"count": len(self.TRAIN), "ids": list(self.TRAIN)},
                "dev": {"count": len(self.DEV), "ids": list(self.DEV)},
                "final_holdout": {"count": len(self.HOLDOUT), "ids": list(self.HOLDOUT)},
            },
            "checksums": {"final_holdout_sha256": id_list_sha256(self.HOLDOUT)},
        }

    def test_id_list_sha256_is_order_independent(self):
        self.assertEqual(id_list_sha256(["b", "a"]), id_list_sha256(["a", "b"]))
        self.assertEqual(
            id_list_sha256(["a"]),
            hashlib.sha256(b"a\n").hexdigest(),
        )

    def test_frozen_partitions_accepted(self):
        good = id_list_sha256(self.HOLDOUT)
        self.assertEqual(assert_partitions_frozen(self.partitions(), good), good)

    def test_unfrozen_constant_refuses(self):
        with self.assertRaises(LeakageGuardError):
            assert_partitions_frozen(self.partitions(), "UNFROZEN")

    def test_mutated_holdout_refuses(self):
        good = id_list_sha256(self.HOLDOUT)
        mutated = self.partitions()
        mutated["partitions"]["final_holdout"]["ids"].pop()
        with self.assertRaises(LeakageGuardError):
            assert_partitions_frozen(mutated, good)

    def test_holdout_id_in_candidates_raises_and_names_it(self):
        with self.assertRaises(LeakageGuardError) as ctx:
            assert_no_holdout(self.TRAIN + [self.HOLDOUT[0]], frozenset(self.HOLDOUT))
        self.assertIn(self.HOLDOUT[0], str(ctx.exception))

    def test_clean_candidates_pass(self):
        assert_no_holdout(self.TRAIN + self.DEV, frozenset(self.HOLDOUT))

    def test_manifest_partition_mismatch_raises(self):
        man = {"suite": {"partitions_sha256": "a" * 64}}
        with self.assertRaises(LeakageGuardError):
            assert_manifest_partitions(man, "b" * 64, "run-x")
        assert_manifest_partitions(man, "a" * 64, "run-x")

    def test_manifest_without_partition_hash_raises(self):
        with self.assertRaises(LeakageGuardError):
            assert_manifest_partitions({"suite": {}}, "a" * 64, "run-x")

    def test_overlapping_partitions_rejected(self):
        import tempfile

        doc = self.partitions()
        doc["partitions"]["train"]["ids"] = list(self.HOLDOUT[:1])
        doc["partitions"]["train"]["count"] = 1
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "partitions.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_partitions(p)

    def test_directory_argument_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ConfigError):
                read_manifest_arg(td)

    def test_trajectory_to_messages(self):
        recs = [
            {"i": 0, "role": "system", "kind": "prompt", "content": "you are an agent"},
            {"i": 1, "role": "user", "kind": "prompt", "content": "fix it"},
            {"i": 2, "role": "assistant", "kind": "tool_call", "tool": "edit_file",
             "args": {"path": "a.py"}},
            {"i": 3, "role": "tool", "kind": "tool_result", "content": "x" * 50},
            {"i": 4, "role": "assistant", "kind": "completion", "content": "done"},
            {"i": 5, "role": "assistant", "kind": "error", "content": "ignored"},
        ]
        msgs = trajectory_messages(recs, max_tool_chars=10)
        self.assertEqual([m["role"] for m in msgs],
                         ["system", "user", "assistant", "tool", "assistant"])
        self.assertIn("edit_file", msgs[2]["content"])
        self.assertIn("truncated", msgs[3]["content"])


def cmd_self_test() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(GuardTests)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    return EXIT_OK if result.wasSuccessful() else EXIT_GUARD


# ------------------------------------------------------------------- main ---


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="build_dataset.py",
        description="Build LoRA training examples from Phase-1 run trajectories.",
        epilog="Inputs are an explicit list of run manifests. Directories are refused.",
    )
    ap.add_argument("--manifests", nargs="+", metavar="RUN_MANIFEST",
                    help="explicit run-manifest.json paths (repeatable list)")
    ap.add_argument("--manifest-list", metavar="FILE",
                    help="file containing one run-manifest.json path per line")
    ap.add_argument("--runs-root", action="append", default=[], metavar="DIR",
                    help="extra root to resolve <run_id>/ under, for manifests kept apart from "
                         "their run dir (e.g. results-index/manifests). Looked up by exact run id.")
    ap.add_argument("--partitions", default=str(repo_root() / "suites" / "partitions.json"),
                    metavar="FILE", help="frozen partitions file (default: suites/partitions.json)")
    ap.add_argument("--split", choices=list(ALLOWED_SPLITS), default="train",
                    help="which partition to draw from (final_holdout is not selectable)")
    ap.add_argument("--out", metavar="DIR",
                    help="output directory (must be outside the repo, or git-ignored)")
    ap.add_argument("--name", default=None, help="dataset base name (default: <split>)")
    ap.add_argument("--include-unresolved", action="store_true",
                    help="also train on attempts that did not resolve (default: resolved only)")
    ap.add_argument("--max-tool-chars", type=int, default=8000,
                    help="truncate each tool result to N chars (0 = never)")
    ap.add_argument("--min-messages", type=int, default=3,
                    help="drop trajectories shorter than N messages")
    ap.add_argument("--max-examples", type=int, default=None, help="cap the dataset size")
    ap.add_argument("--no-verify-sha", action="store_true",
                    help="skip trajectory sha256 verification (faster, less safe)")
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing dataset-manifest.json in --out")
    ap.add_argument("--freeze", metavar="PARTITIONS",
                    help="print the FINAL_HOLDOUT_SHA256 line for a frozen partitions.json")
    ap.add_argument("--write", action="store_true", help="with --freeze: patch this file in place")
    ap.add_argument("--print-holdout-constant", action="store_true",
                    help="print the compiled-in FINAL_HOLDOUT_SHA256 and exit")
    ap.add_argument("--self-test", action="store_true", help="run the leakage-guard unit tests")
    return ap.parse_args(argv)


def run(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if args.print_holdout_constant:
        print(FINAL_HOLDOUT_SHA256)
        return EXIT_OK
    if args.self_test:
        return cmd_self_test()
    if args.freeze:
        return cmd_freeze(Path(args.freeze).expanduser(), args.write)

    if bool(args.manifests) == bool(args.manifest_list):
        raise ConfigError("give exactly one of --manifests or --manifest-list")
    if not args.out and not args.dry_run:
        raise ConfigError("--out is required (or use --dry-run)")

    manifest_paths = (
        [read_manifest_arg(a) for a in args.manifests]
        if args.manifests
        else read_manifest_list(Path(args.manifest_list).expanduser())
    )

    partitions_path = Path(args.partitions).expanduser()
    partitions = load_partitions(partitions_path)
    partitions_sha256 = sha256_path(partitions_path)
    holdout_sha256 = assert_partitions_frozen(partitions)
    holdout = holdout_ids(partitions)
    wanted = split_ids(partitions, args.split)
    # Belt and braces: the split we are about to train on must be disjoint from the holdout.
    assert_no_holdout(wanted, holdout)

    runs = load_source_runs(manifest_paths, [Path(r).expanduser() for r in args.runs_root])
    for r in runs:
        assert_manifest_partitions(r.manifest, partitions_sha256, f"{r.run_id} ({r.manifest_path})")

    examples, stats = build_examples(
        runs,
        wanted_ids=wanted,
        holdout=holdout,
        part_index=partition_index(partitions),
        split=args.split,
        require_resolved=not args.include_unresolved,
        max_tool_chars=args.max_tool_chars,
        min_messages=args.min_messages,
        verify_sha=not args.no_verify_sha,
        max_examples=args.max_examples,
    )
    # Final assertion over what we actually built — guard 3 once more, on the output.
    assert_no_holdout((ex["qualified_id"] for ex in examples), holdout)

    print(
        f"==> {stats.kept} examples from {len(runs)} run(s); considered {stats.considered} "
        f"attempts; skipped {stats.skipped_wrong_split} out-of-split, "
        f"{stats.skipped_unresolved} unresolved, {stats.skipped_no_trajectory} without a "
        f"trajectory",
        file=sys.stderr,
    )
    if args.dry_run:
        print(json.dumps({"split": args.split, "stats": stats.as_dict()}, sort_keys=True))
        return EXIT_OK
    if not examples:
        raise ConfigError("no examples survived the filters — refusing to write an empty dataset")

    out_dir = Path(args.out).expanduser().resolve()
    assert_writable_outside_git(out_dir)
    data_path, manifest_path = write_dataset(
        out_dir,
        args.name or args.split,
        examples,
        runs=runs,
        partitions_path=partitions_path,
        partitions_sha256=partitions_sha256,
        holdout_sha256=holdout_sha256,
        split=args.split,
        stats=stats,
        filters={
            "require_resolved": not args.include_unresolved,
            "max_tool_chars": args.max_tool_chars,
            "min_messages": args.min_messages,
            "max_examples": args.max_examples,
            "verify_trajectory_sha256": not args.no_verify_sha,
        },
        force=args.force,
    )
    print(f"==> wrote {manifest_path}", file=sys.stderr)
    print(f"{data_path} {sha256_path(data_path)} {len(examples)}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(list(sys.argv[1:] if argv is None else argv))
    except LeakageGuardError as exc:
        print(f"LEAKAGE GUARD: {exc}", file=sys.stderr)
        return EXIT_GUARD
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
