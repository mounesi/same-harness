"""Shared implementation behind the two SWE-bench adapters.

`swebench_verified.py` and `swebench_pro.py` are thin modules: each declares a
`SuiteSpec` and delegates every contract function here.  The two suites share one
code path because their task record shape and their grading contract agree; they
differ only in the fields captured by `SuiteSpec` (dataset, seed file, grader
label, container-image resolution) and in nothing else.

Contract: docs/CONTRACTS.md §5 (adapter interface), §4 (failure taxonomy),
§6.1 (suite seed files).  This module is NOT an adapter — it declares no
SUITE_NAME and must never be registered in `harness.adapters.ADAPTERS`.

Three things a reader should know before changing anything here.

1. The seed file is authoritative.  `load_tasks` reads the instance ids from
   `suites/verified-100.json` / `suites/pro-50.json` and never samples, filters,
   re-orders or tops up.  A dataset row that is missing for a seeded id is a
   configuration error, not something to route around.

2. Benchmark data is never vendored.  Rows are pulled at run time through the
   HuggingFace `datasets` library, pinned to `source.revision` from the seed
   file.  Per §5, `load_tasks` does not reach the network on its own: it runs the
   library in offline mode and reads the local HF cache.  Set
   `HARNESS_ALLOW_NETWORK=1` (the same gate §2.2 uses for HfApi revision lookup)
   to permit the download, or prefetch once with
   `python3 -m harness.adapters.swebench_verified --prefetch`.

3. Grading shells out to the official SWE-bench evaluation harness.  See
   `grade()` for the exact invocation and the report parsing.  The grader is
   handed the *pinned* dataset row (the one `load_tasks` built the task from),
   never a bare hub id: the hub id would make upstream resolve the dataset
   HEAD, silently un-pinning the run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The adapters are imported both as `harness.adapters.<suite>` (from run.sh /
# agent.py) and directly as a script (`python3 -m harness.adapters.swebench_pro`).
# Guarantee the repo root is importable so `harness.types` / `harness.prompts`
# resolve either way.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.types import GraderError, Prompt, Task, Verdict  # noqa: E402

# dataclass(slots=True) needs CPython >= 3.10. Target runtime is 3.11 (Lambda images);
# degrade on older interpreters so the repo stays importable for local inspection.
_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}

__all__ = [
    "SuiteSpec",
    "AdapterConfigError",
    "DependencyMissing",
    "DatasetUnavailable",
    "SeedMismatch",
    "load_tasks",
    "build_prompt",
    "grade",
    "environment_digest",
    "main",
]

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

SEED_SCHEMA = "suite-seed/v1"
PARTITIONS_SCHEMA = "partitions/v1"
INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
VALID_PARTITIONS = ("train", "dev", "final_holdout", "unpartitioned")
DETAIL_MAX = 512

#: `model_name_or_path` written into the predictions file handed to the official
#: evaluation harness.  Constant on purpose: it is part of the report filename,
#: and using the real model name would make grader output vary per model.
PREDICTION_TAG = "agenttask-harness"

#: Wall-clock ceiling for one `grade()` subprocess.  This is grader
#: infrastructure, unrelated to the agent's `--task-timeout` budget.
DEFAULT_GRADER_TIMEOUT_S = 3600

#: Per-test timeout passed through to the evaluation harness.
DEFAULT_TEST_TIMEOUT_S = 1800

#: Last-resort test command used only for the `test_cmd` prompt variable when
#: neither the dataset row nor the installed swebench constants supply one.
#: Identical for both suites, so it cannot make the harness vary by suite.
FALLBACK_TEST_CMD = "python -m pytest"

#: Upstream `swebench.harness.constants.APPLY_PATCH_FAIL` as it appears in
#: `run_instance.log` (lower-cased for matching).  Its presence is definitive:
#: the container was built and the model's patch did not apply.
_PATCH_APPLY_FAIL_MARKERS = (">>>>> patch apply failed",)

#: Id lists written by upstream `make_run_report` into the run summary
#: `<tag>.<run_id>.json`.  The `*_instances` keys next to them are INTEGER
#: COUNTS, never membership lists — do not test `instance_id in` against them.
_SUMMARY_ID_LISTS = (
    "resolved_ids",
    "unresolved_ids",
    "error_ids",
    "empty_patch_ids",
    "completed_ids",
    "submitted_ids",
)

#: Upper bound on grader log text folded into the sandbox / apply-fail checks.
_LOG_EVIDENCE_MAX = 200_000

_SANDBOX_MARKERS = (
    "cannot connect to the docker daemon",
    "docker daemon",
    "pull access denied",
    "no such image",
    "error building",
    "failed to build",
    "image not found",
    "manifest unknown",
    "no space left on device",
    "toomanyrequests",
)


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


class AdapterConfigError(ValueError):
    """Bad seed file / partitions file / adapter configuration.

    run.sh maps this to exit code 2 (config)."""


class DependencyMissing(AdapterConfigError):
    """A required third-party package is not installed."""


class DatasetUnavailable(AdapterConfigError):
    """Task rows could not be obtained (no cache and no network permission)."""


class SeedMismatch(AdapterConfigError):
    """The pinned dataset does not contain every seeded instance id."""


# --------------------------------------------------------------------------- #
# suite specification — the only thing the two adapters disagree about
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True, **_SLOTS)
class SuiteSpec:
    """Everything that differs between swebench-verified and swebench-pro."""

    suite_name: str
    adapter_version: str
    consent_class: str
    #: HuggingFace dataset id expected in the seed file's `source.dataset`.
    dataset: str
    #: Dataset split expected in the seed file's `source.split`.
    split: str
    #: Repo-relative default seed file (informational; run.sh passes the path).
    default_seed_file: str
    #: `Verdict.grader` label recorded on every attempt.
    grader: str
    #: python -m target of the official evaluation harness.
    eval_module: str
    #: Distribution whose version identifies the grader (for grader_version and
    #: environment_digest).
    grader_distribution: str
    #: Environment-variable infix, e.g. "SWEBENCH_VERIFIED".
    env_infix: str
    #: Container image template, `{instance_id}` / `{norm_id}` substituted.
    #: None means "the evaluation harness resolves the image itself".
    image_template: str | None
    #: Dataset-row keys that may carry an explicit image name.
    image_row_keys: tuple[str, ...] = ()

    def env(self, key: str, default: str | None = None) -> str | None:
        """Read `HARNESS_<KEY>_<SUITE>` falling back to `HARNESS_<KEY>`."""
        specific = os.environ.get(f"HARNESS_{key}_{self.env_infix}")
        if specific is not None:
            return specific
        return os.environ.get(f"HARNESS_{key}", default)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON per CONTRACTS §0 (compact, sorted keys, UTF-8)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_digest(ids: Iterable[str]) -> str:
    """CONTRACTS §6.1: sha256("\\n".join(sorted(ids)) + "\\n")."""
    joined = "\n".join(sorted(ids)) + "\n"
    return _sha256_hex(joined.encode("utf-8"))


def _clip(text: str, limit: int = DETAIL_MAX) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _read_json(path: Path, what: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AdapterConfigError(f"{what}: cannot read {path}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterConfigError(f"{what}: {path} is not valid JSON: {exc}") from exc


def _as_str_tuple(value: Any, field: str, instance_id: str) -> tuple[str, ...]:
    """SWE-bench stores FAIL_TO_PASS/PASS_TO_PASS as a JSON-encoded list.

    Some mirrors store a real list.  Accept both, reject anything else.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AdapterConfigError(
                f"{instance_id}: {field} is a string but not JSON: {exc}"
            ) from exc
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise AdapterConfigError(f"{instance_id}: {field} has unsupported type {type(value).__name__}")


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _network_allowed() -> bool:
    return os.environ.get("HARNESS_ALLOW_NETWORK", "0") == "1"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise AdapterConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _dist_version(distribution: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(distribution)
    except Exception:
        return None


def _dist_record_sha256(distribution: str) -> str | None:
    """sha256 of the distribution's RECORD file.

    Pins the installed artifact rather than the version string, mirroring how
    CONTRACTS §2.2 pins vLLM.
    """
    try:
        import importlib.metadata as md

        dist = md.distribution(distribution)
        record = dist.read_text("RECORD")
    except Exception:
        return None
    if record is None:
        return None
    return _sha256_hex(record.encode("utf-8"))


# --------------------------------------------------------------------------- #
# seed file — authoritative instance list
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True, **_SLOTS)
class Seed:
    path: Path
    suite: str
    dataset: str
    revision: str | None
    split: str
    instance_ids: tuple[str, ...]
    sha256: str
    raw: dict


def load_seed(spec: SuiteSpec, seed_file: Path) -> Seed:
    """Parse and validate a `suite-seed/v1` file.  Every failure is a config error."""
    seed_file = Path(seed_file)
    doc = _read_json(seed_file, "seed file")
    if not isinstance(doc, dict):
        raise AdapterConfigError(f"seed file {seed_file}: top level must be an object")

    schema = doc.get("schema")
    if schema != SEED_SCHEMA:
        raise AdapterConfigError(
            f"seed file {seed_file}: schema is {schema!r}, expected {SEED_SCHEMA!r}"
        )
    if doc.get("suite") != spec.suite_name:
        raise AdapterConfigError(
            f"seed file {seed_file}: suite is {doc.get('suite')!r}, "
            f"expected {spec.suite_name!r} — wrong seed file for this adapter"
        )

    ids = doc.get("instance_ids")
    if not isinstance(ids, list) or not ids:
        raise AdapterConfigError(f"seed file {seed_file}: instance_ids must be a non-empty list")
    if not all(isinstance(i, str) for i in ids):
        raise AdapterConfigError(f"seed file {seed_file}: instance_ids must be strings")
    bad = [i for i in ids if not INSTANCE_ID_RE.match(i)]
    if bad:
        raise AdapterConfigError(
            f"seed file {seed_file}: instance ids must match {INSTANCE_ID_RE.pattern} "
            f"(they become directory names); offenders: {bad[:5]}"
        )
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise AdapterConfigError(f"seed file {seed_file}: duplicate instance ids: {dupes[:5]}")

    count = doc.get("count")
    if count is not None and count != len(ids):
        raise AdapterConfigError(
            f"seed file {seed_file}: count is {count} but instance_ids has {len(ids)} entries"
        )

    declared = doc.get("instance_ids_sha256")
    computed = ids_digest(ids)
    if not declared and not doc.get("placeholder"):
        # Mirror harness/manifest.py: a non-placeholder seed without its seal is not a
        # frozen seed (§6.1) — an absent hash must not pass what a wrong one fails.
        raise AdapterConfigError(
            f"seed file {seed_file}: instance_ids_sha256 is missing — a frozen seed file "
            "MUST carry the seal of its id set (§6.1); regenerate it with suites/select.py"
        )
    if declared and declared != computed:
        raise AdapterConfigError(
            f"seed file {seed_file}: instance_ids_sha256 mismatch — "
            f"declared {declared}, computed {computed}. The seed file has been edited "
            f"since it was frozen; regenerate it with suites/select.py or restore it."
        )

    source = doc.get("source") or {}
    if not isinstance(source, dict):
        raise AdapterConfigError(f"seed file {seed_file}: source must be an object")
    dataset = source.get("dataset") or spec.dataset
    if dataset != spec.dataset:
        raise AdapterConfigError(
            f"seed file {seed_file}: source.dataset is {dataset!r}, "
            f"but this adapter is pinned to {spec.dataset!r}"
        )
    split = source.get("split") or spec.split
    revision = _checked_revision(seed_file, source.get("revision") or None)

    return Seed(
        path=seed_file,
        suite=spec.suite_name,
        dataset=dataset,
        revision=revision,
        split=split,
        instance_ids=tuple(ids),
        sha256=_sha256_hex(seed_file.read_bytes()),
        raw=doc,
    )


_UNRESOLVED_REVISION_MARKERS = ("todo", "placeholder", "unresolved", "xxx", "fixme")

#: `environment_digest()` payload value when the seed file pins no dataset
#: revision (HARNESS_ALLOW_UNPINNED_DATASET=1 runs against the dataset head).
UNPINNED_REVISION = "unpinned"


def _looks_unresolved(revision: str | None) -> bool:
    if revision is None or not str(revision).strip():
        return True
    lowered = str(revision).lower()
    return any(m in lowered for m in _UNRESOLVED_REVISION_MARKERS)


def _checked_revision(seed_file: Path, revision: str | None) -> str | None:
    """Reject a seed file whose dataset revision is still a placeholder.

    An unpinned dataset silently changes what "the same 100 instances" means, so
    this is a hard configuration error rather than a warning. Set
    HARNESS_ALLOW_UNPINNED_DATASET=1 to proceed against the dataset head while
    the pin is being resolved; the run is not publication-grade.
    """
    if not _looks_unresolved(revision):
        return revision
    if os.environ.get("HARNESS_ALLOW_UNPINNED_DATASET") == "1":
        print(
            f"==> WARNING: {seed_file} has no resolved source.revision "
            f"({revision!r}); loading the dataset head. This run is not reproducible.",
            file=sys.stderr,
        )
        return None
    raise AdapterConfigError(
        f"seed file {seed_file}: source.revision is {revision!r}, which is not a resolved "
        "dataset revision. Pin it to the dataset commit the subset was drawn from "
        "(the seed file is the reproducibility record), or set "
        "HARNESS_ALLOW_UNPINNED_DATASET=1 to run against the dataset head."
    )


# --------------------------------------------------------------------------- #
# partitions
# --------------------------------------------------------------------------- #


def _default_partitions_path() -> Path | None:
    override = os.environ.get("HARNESS_PARTITIONS")
    if override:
        return Path(override)
    candidate = _REPO_ROOT / "suites" / "partitions.json"
    return candidate if candidate.exists() else None


def load_partition_map(partitions_file: Path | None) -> dict[str, str]:
    """qualified_id -> partition name.  Missing file means everything is unpartitioned."""
    if partitions_file is None:
        return {}
    path = Path(partitions_file)
    if not path.exists():
        raise AdapterConfigError(f"partitions file not found: {path}")
    doc = _read_json(path, "partitions file")
    if not isinstance(doc, dict):
        raise AdapterConfigError(f"partitions file {path}: top level must be an object")
    if doc.get("schema") != PARTITIONS_SCHEMA:
        raise AdapterConfigError(
            f"partitions file {path}: schema is {doc.get('schema')!r}, "
            f"expected {PARTITIONS_SCHEMA!r}"
        )
    partitions = doc.get("partitions")
    if not isinstance(partitions, dict):
        raise AdapterConfigError(f"partitions file {path}: partitions must be an object")

    mapping: dict[str, str] = {}
    for name, block in partitions.items():
        if name not in VALID_PARTITIONS:
            raise AdapterConfigError(
                f"partitions file {path}: unknown partition {name!r}, "
                f"expected one of {VALID_PARTITIONS}"
            )
        ids = (block or {}).get("ids") if isinstance(block, dict) else None
        if not isinstance(ids, list):
            raise AdapterConfigError(f"partitions file {path}: partitions.{name}.ids must be a list")
        for qualified in ids:
            if qualified in mapping and mapping[qualified] != name:
                raise AdapterConfigError(
                    f"partitions file {path}: {qualified} appears in both "
                    f"{mapping[qualified]!r} and {name!r}"
                )
            mapping[str(qualified)] = name
    return mapping


# --------------------------------------------------------------------------- #
# dataset rows — fetched at run time, never vendored
# --------------------------------------------------------------------------- #

_ROW_CACHE: dict[tuple[str, str, str | None], dict[str, dict]] = {}


def _import_datasets(offline: bool):
    """Import `datasets`, with an actionable error when it is absent.

    The offline switches must be in the environment before the library reads its
    config, so they are set here rather than at module import time.
    """
    if offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import datasets  # type: ignore
    except ImportError as exc:
        raise DependencyMissing(
            "the 'datasets' package is required to load SWE-bench task rows and is not "
            "installed. Benchmark data is deliberately not vendored into this repo. "
            "Install the pinned harness dependencies:\n"
            "    python3 -m pip install -r harness/requirements.txt\n"
            "(or, for the exact study environment, harness/requirements.lock)."
        ) from exc
    return datasets


def _prefetch_hint(spec: SuiteSpec) -> str:
    return (
        "Prefetch once on this host with\n"
        f"    HARNESS_ALLOW_NETWORK=1 python3 -m harness.adapters."
        f"{spec.suite_name.replace('-', '_')} --prefetch\n"
        "or export HARNESS_ALLOW_NETWORK=1 for this run."
    )


def _rows_for(spec: SuiteSpec, dataset: str, split: str, revision: str | None) -> dict[str, dict]:
    """{instance_id: row} for one (dataset, split, revision), memoised in `_ROW_CACHE`.

    Honours the §5 network gate: offline unless HARNESS_ALLOW_NETWORK=1, so a
    miss with no local HF cache raises `DatasetUnavailable` instead of
    downloading.  `revision=None` means the caller explicitly accepted the
    dataset head (HARNESS_ALLOW_UNPINNED_DATASET=1); it is part of the cache key
    so rows from different revisions are never conflated.
    """
    key = (dataset, split, revision)
    cached = _ROW_CACHE.get(key)
    if cached is not None:
        return cached
    offline = not _network_allowed()
    datasets = _import_datasets(offline)
    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    try:
        ds = datasets.load_dataset(dataset, **kwargs)
    except Exception as exc:  # datasets raises a wide variety of types
        hint = (
            "No local HuggingFace cache entry and network access is disabled. " + _prefetch_hint(spec)
            if offline
            else "Network access was permitted but the download failed."
        )
        raise DatasetUnavailable(
            f"could not load {dataset} (split={split}, "
            f"revision={revision or UNPINNED_REVISION}): {exc}. {hint}"
        ) from exc
    cached = {}
    for row in ds:
        instance_id = row.get("instance_id")
        if isinstance(instance_id, str):
            cached[instance_id] = dict(row)
    _ROW_CACHE[key] = cached
    return cached


def _load_rows(spec: SuiteSpec, seed: Seed) -> dict[str, dict]:
    """Return {instance_id: row} for the seeded ids only, pinned to the seed revision."""
    cached = _rows_for(spec, seed.dataset, seed.split, seed.revision)

    missing = [i for i in seed.instance_ids if i not in cached]
    if missing:
        raise SeedMismatch(
            f"{len(missing)} seeded instance id(s) are absent from {seed.dataset} "
            f"(split={seed.split}, revision={seed.revision or 'unpinned'}): "
            f"{missing[:5]}{' …' if len(missing) > 5 else ''}. "
            "The seed file is authoritative and is never silently trimmed — either the "
            "pinned revision is wrong or the seed file was generated against a different "
            "population."
        )
    return {i: cached[i] for i in seed.instance_ids}


# --------------------------------------------------------------------------- #
# Task construction
# --------------------------------------------------------------------------- #


def _normalized_image_id(instance_id: str) -> str:
    """SWE-bench container-image id normalisation (double underscore is illegal)."""
    return instance_id.lower().replace("__", "_1776_")


def _resolve_image(spec: SuiteSpec, row: Mapping[str, Any], instance_id: str) -> str:
    override = spec.env("IMAGE_TEMPLATE")
    template = override if override else spec.image_template
    from_row = _first_present(row, spec.image_row_keys) if spec.image_row_keys else None
    if from_row:
        return str(from_row)
    if not template:
        return ""
    return template.format(instance_id=instance_id, norm_id=_normalized_image_id(instance_id))


def _resolve_test_cmd(row: Mapping[str, Any], repo: str) -> str:
    """Best available per-instance test command, used for the `test_cmd` prompt variable.

    Order: explicit dataset column -> the installed swebench constants table ->
    a constant fallback.  The same order is used for both suites, so this cannot
    become a source of per-suite prompt drift.
    """
    explicit = _first_present(row, ("test_cmd", "test_command", "run_tests"))
    if explicit:
        return str(explicit)

    version = row.get("version")
    if repo and version is not None:
        try:
            from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # type: ignore

            candidate = MAP_REPO_VERSION_TO_SPECS[repo][str(version)].get("test_cmd")
            if isinstance(candidate, str) and candidate:
                return candidate
        except Exception:
            pass

    directives = _first_present(row, ("test_directives",))
    if directives:
        try:
            items = _as_str_tuple(directives, "test_directives", str(row.get("instance_id")))
            if items:
                return f"{FALLBACK_TEST_CMD} {' '.join(items)}"
        except AdapterConfigError:
            pass
    return FALLBACK_TEST_CMD


def _build_task(
    spec: SuiteSpec,
    row: Mapping[str, Any],
    instance_id: str,
    partitions: Mapping[str, str],
    revision: str | None,
) -> Task:
    repo = str(_first_present(row, ("repo", "repository")) or "")
    base_commit = str(_first_present(row, ("base_commit", "commit")) or "")
    problem = _first_present(row, ("problem_statement", "issue_text", "problem"))
    if not problem:
        raise AdapterConfigError(f"{instance_id}: dataset row has no problem statement")

    fail_to_pass = _as_str_tuple(
        _first_present(row, ("FAIL_TO_PASS", "fail_to_pass")), "FAIL_TO_PASS", instance_id
    )
    pass_to_pass = _as_str_tuple(
        _first_present(row, ("PASS_TO_PASS", "pass_to_pass")), "PASS_TO_PASS", instance_id
    )
    if not fail_to_pass:
        raise AdapterConfigError(
            f"{instance_id}: FAIL_TO_PASS is empty — the instance has no success criterion"
        )

    qualified_id = f"{spec.suite_name}::{instance_id}"
    test_cmd = _resolve_test_cmd(row, repo)
    environment = {
        "image": _resolve_image(spec, row, instance_id),
        # SWE-bench evaluation images ship the environment prebuilt; the official
        # harness performs setup itself. Kept for shape-compatibility with the
        # agenttask adapter.
        "setup_cmds": [],
        "test_cmd": test_cmd,
    }

    # `metadata` is suite-specific and is never fed to the prompt (CONTRACTS §5.1).
    # `hints_text` is deliberately dropped, not stored: it is a leak of the gold
    # solution and must not be reachable from the agent.
    # `revision` is the seed's pinned `source.revision` (None only under
    # HARNESS_ALLOW_UNPINNED_DATASET=1).  grade() uses it to hand the evaluation
    # harness exactly the row this task was built from — see `_pinned_row`.
    metadata = {
        "dataset": spec.dataset,
        "split": spec.split,
        "revision": revision,
        "version": row.get("version"),
        "environment_setup_commit": row.get("environment_setup_commit"),
        "test_patch_present": bool(row.get("test_patch")),
        "created_at": row.get("created_at"),
        "hints_dropped": bool(row.get("hints_text")),
    }

    return Task(
        suite=spec.suite_name,
        instance_id=instance_id,
        qualified_id=qualified_id,
        repo=repo,
        base_commit=base_commit,
        problem_statement=str(problem),
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        environment=environment,
        partition=partitions.get(qualified_id, "unpartitioned"),
        metadata=metadata,
        source_sha256=_sha256_hex(_canonical_bytes(_jsonable(row))),
    )


def _jsonable(row: Mapping[str, Any]) -> dict:
    """Coerce a dataset row to something json.dumps can canonicalise."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        elif isinstance(value, (list, tuple)):
            out[str(key)] = [v if isinstance(v, (str, int, float, bool)) or v is None else str(v) for v in value]
        elif isinstance(value, dict):
            out[str(key)] = {str(k): (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in value.items()}
        else:
            out[str(key)] = str(value)
    return out


def load_tasks(
    spec: SuiteSpec,
    seed_file: Path,
    partitions_file: Path | None = None,
) -> list[Task]:
    """Contract entry point.  Returns tasks in seed-file order, one per seeded id.

    `partitions_file` is an optional extension over the §5 signature: when it is
    omitted the adapter falls back to `$HARNESS_PARTITIONS`, then to
    `suites/partitions.json`, then to `partition="unpartitioned"`.
    """
    seed = load_seed(spec, Path(seed_file))
    partitions = load_partition_map(
        partitions_file if partitions_file is not None else _default_partitions_path()
    )
    rows = _load_rows(spec, seed)
    return [_build_task(spec, rows[i], i, partitions, seed.revision) for i in seed.instance_ids]


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #


def template_id() -> str:
    """The single template id shared by every suite (CONTRACTS §5.2)."""
    try:
        from harness import prompts as harness_prompts

        tid = getattr(harness_prompts, "TEMPLATE_ID", None)
        if tid:
            return str(tid)
    except ImportError:
        pass
    path = _REPO_ROOT / "harness" / "prompts" / "TEMPLATE_ID"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AdapterConfigError(
            f"prompt template id not readable at {path}: {exc}. "
            "harness/prompts/ is required; see CONTRACTS §5.2."
        ) from exc


def build_prompt(spec: SuiteSpec, task: Task) -> Prompt:
    """Render the shared template.  Identical call in every adapter, by contract.

    The harness is the control variable: both SWE-bench adapters pass the same
    template id and the same variable names; only the values differ.
    """
    from harness import prompts as harness_prompts

    tid = template_id()
    prompt = harness_prompts.render(
        tid,
        {
            "problem_statement": task.problem_statement,
            "repo": task.repo,
            "test_cmd": task.environment.get("test_cmd", ""),
        },
    )
    if prompt.template_id != tid:
        raise AdapterConfigError(
            f"{spec.suite_name}: prompt template id drift — renderer returned "
            f"{prompt.template_id!r}, harness/prompts/TEMPLATE_ID says {tid!r}"
        )
    return prompt


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
#
# HOW EVALUATION IS INVOKED
# -------------------------
# `grade()` shells out to the **official SWE-bench evaluation harness**
# (`python3 -m swebench.harness.run_evaluation`) as a subprocess, one instance
# per call, with `--max_workers 1`.  We do not re-implement test running or log
# parsing: the upstream harness builds/pulls the instance's evaluation image,
# applies our model patch inside the container, applies the instance's own
# `test_patch`, runs FAIL_TO_PASS + PASS_TO_PASS, and writes a machine-readable
# report.  That report is the sole source of the verdict.
#
# Inputs written into a private temp dir (nothing is written outside it):
#   dataset.jsonl  the single pinned dataset row — the exact row `load_tasks`
#                  built the task from, at the seed file's `source.revision`
#                  (`--dataset_name` accepts a local .jsonl path).  The bare hub
#                  id is NEVER passed: `run_evaluation` has no revision flag, so
#                  a hub id makes it grade against the dataset HEAD, silently
#                  un-pinning the run.  If the pinned rows are not cached
#                  in-process or in the local HF cache, grade() raises
#                  GraderError naming `--prefetch` (see `_pinned_row`).
#   preds.json     [{"instance_id", "model_name_or_path": PREDICTION_TAG,
#                    "model_patch": <patch>}]
#
# The subprocess runs with cwd=<temp dir>, so the harness's `logs/` tree and its
# summary report land inside the temp dir and are removed with it.
#
# Image cache: `--cache_level env --clean False`.  Upstream keeps the base and
# per-environment (conda/pip) images and removes only the per-instance image
# after the run.  With `--cache_level none` every one of the ~900 grade() calls
# in a study — each a fresh subprocess, run_id and temp dir — rebuilt the
# environment image from scratch (5-15 min of the ~$54/hr node per instance).
# Caching the env layer changes nothing about the verdict: environment_digest()
# hashes the swebench distribution (version + RECORD) and the docker server
# version, i.e. the *recipe*, not the image cache, so reproducibility is
# unaffected.  `--clean False` keeps images that existed before the run.
#
# Report parsing: the per-instance report.json
# (logs/run_evaluation/<run_id>/<tag>/<id>/report.json, or any nested
# report.json) is the only source of a scored verdict.  The run summary
# `<tag>.<run_id>.json` (written by upstream `make_run_report`) is consulted
# only when no per-instance report exists, and then it can only yield NO_PATCH,
# PATCH_MALFORMED, INFRA_SANDBOX or GraderError — never TESTS_FAIL, because an
# instance whose container never built also lands in the summary's `error_ids`.
#
# Verdict mapping (CONTRACTS §4):
#   patch empty/whitespace                -> NO_PATCH   (no environment built)
#   patch did not apply                   -> PATCH_MALFORMED
#   resolved                              -> OK, resolved=True
#   all FAIL_TO_PASS pass, PASS_TO_PASS regressed -> TESTS_REGRESSION
#   otherwise                             -> TESTS_FAIL
#   docker/image/setup failure            -> INFRA_SANDBOX (returned, not raised)
#   grader crash / timeout / no report /
#   report without a parseable tests_status -> GraderError -> INFRA_GRADER
#
# Both suites use this identical path.  swebench-pro differs only in
# `spec.dataset`, `spec.grader` and `spec.eval_module`; if the Pro evaluation
# entry point diverges upstream, point HARNESS_EVAL_CMD_SWEBENCH_PRO at it
# instead of forking this function.


def _eval_command(
    spec: SuiteSpec,
    dataset_arg: str,
    predictions: Path,
    instance_id: str,
    run_tag: str,
    report_dir: Path,
    test_timeout: int,
) -> list[str]:
    fields = {
        "python": sys.executable or "python3",
        "module": spec.eval_module,
        "dataset": dataset_arg,
        "split": spec.split,
        "predictions": str(predictions),
        "instance_id": instance_id,
        "run_id": run_tag,
        "report_dir": str(report_dir),
        "timeout": str(test_timeout),
    }
    override = spec.env("EVAL_CMD")
    if override:
        return [part.format(**fields) for part in shlex.split(override)]
    return [
        fields["python"],
        "-m",
        spec.eval_module,
        "--dataset_name",
        dataset_arg,
        "--split",
        spec.split,
        "--predictions_path",
        str(predictions),
        "--instance_ids",
        instance_id,
        "--run_id",
        run_tag,
        "--max_workers",
        "1",
        # Keep base+env images, drop the instance image (see the block comment
        # above for why `none` was a 5-15 min rebuild per grade() call).
        "--cache_level",
        "env",
        "--clean",
        "False",
        "--timeout",
        str(test_timeout),
    ]


def _find_report(tmp: Path, instance_id: str) -> tuple[dict | None, dict | None]:
    """Locate the grader's output for `instance_id` under `tmp`.

    Returns `(instance_report, summary)`:
      * `instance_report` — the body of the per-instance report.json (upstream
        writes `{instance_id: {...}}`), the only source of a scored verdict;
      * `summary` — the run-level `<tag>.<run_id>.json` from `make_run_report`,
        recognised by its `*_ids` membership lists.
    Either may be None.  A file that is unreadable or not JSON (e.g. truncated
    mid-write) is skipped, and the caller treats a missing per-instance report
    as INFRA_* / GraderError — never as a scored TESTS_FAIL.
    """
    instance_report: dict | None = None
    summary: dict | None = None
    candidates: list[Path] = sorted(tmp.rglob("report.json"))
    candidates.extend(sorted(p for p in tmp.glob("*.json") if p.name != "preds.json"))
    for path in candidates:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        if instance_report is None:
            if instance_id in doc and isinstance(doc[instance_id], dict):
                instance_report = doc[instance_id]
                continue
            if "tests_status" in doc or "patch_successfully_applied" in doc:
                instance_report = doc
                continue
        if summary is None and any(isinstance(doc.get(k), list) for k in _SUMMARY_ID_LISTS):
            summary = doc
    return instance_report, summary


def _in_summary_list(summary: Mapping[str, Any], key: str, instance_id: str) -> bool:
    """Membership in one of upstream's `*_ids` lists (the `*_instances` keys are counts)."""
    ids = summary.get(key)
    return isinstance(ids, list) and instance_id in ids


def _counts(block: Any, label: str, instance_id: str, expected_total: int) -> dict:
    """{"passed", "total"} from one `tests_status.<label>` block.

    Raises GraderError (-> INFRA_GRADER) when the block is missing or malformed:
    a verdict we cannot read must not be scored as 0/N.  `total` is what the
    grader actually reported — an empty block is 0/0, never silently 0/N.
    """
    if (
        not isinstance(block, dict)
        or not isinstance(block.get("success"), list)
        or not isinstance(block.get("failure"), list)
    ):
        raise GraderError(
            f"unparseable verdict for {instance_id}: tests_status.{label} is missing or "
            f"malformed (got {type(block).__name__}); the evaluation harness report "
            "format may have changed"
        )
    passed = len(block["success"])
    total = passed + len(block["failure"])
    if total == 0 and expected_total > 0:
        raise GraderError(
            f"unparseable verdict for {instance_id}: tests_status.{label} lists no "
            f"outcomes although the task has {expected_total} {label} test(s) — the "
            "grader's log parser produced nothing"
        )
    return {"passed": passed, "total": total}


def _looks_like_sandbox_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in _SANDBOX_MARKERS)


def _looks_like_patch_apply_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in _PATCH_APPLY_FAIL_MARKERS)


def _instance_log_text(tmp: Path, instance_id: str) -> str:
    """Tail of every grader log that belongs to `instance_id` (run_instance.log,
    build_image.log), capped at `_LOG_EVIDENCE_MAX` bytes in total."""
    needles = (instance_id.lower(), _normalized_image_id(instance_id))
    parts: list[str] = []
    budget = _LOG_EVIDENCE_MAX
    for path in sorted(tmp.rglob("*.log")):
        if budget <= 0:
            break
        lowered_path = str(path).lower()
        if not any(n in lowered_path for n in needles):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunk = text[-budget:]
        budget -= len(chunk)
        parts.append(f"===== {path.relative_to(tmp)} =====\n{chunk}")
    return "\n".join(parts)


def grade(spec: SuiteSpec, task: Task, patch: str) -> Verdict:
    """Run the instance's tests against `patch` and return a Verdict.

    Never raises for a task-level failure; raises GraderError only when the
    grading infrastructure itself failed (mapped to INFRA_GRADER by the caller).
    """
    grader_version = _dist_version(spec.grader_distribution) or "unknown"

    if not patch or not patch.strip():
        return Verdict(
            resolved=False,
            error_code="NO_PATCH",
            detail="empty patch; no evaluation environment was built",
            fail_to_pass={"passed": 0, "total": len(task.fail_to_pass)},
            pass_to_pass={"passed": 0, "total": len(task.pass_to_pass)},
            grader=spec.grader,
            grader_version=grader_version,
            raw={"skipped": "empty_patch"},
        )

    if shutil.which("docker") is None:
        return Verdict(
            resolved=False,
            error_code="INFRA_SANDBOX",
            detail=(
                "docker is not available on PATH; the SWE-bench evaluation harness "
                "requires it to build the instance container"
            ),
            fail_to_pass={"passed": 0, "total": len(task.fail_to_pass)},
            pass_to_pass={"passed": 0, "total": len(task.pass_to_pass)},
            grader=spec.grader,
            grader_version=grader_version,
            raw={"error": "docker_missing"},
        )

    grader_timeout = _int_env("HARNESS_GRADER_TIMEOUT", DEFAULT_GRADER_TIMEOUT_S)
    test_timeout = _int_env("HARNESS_TEST_TIMEOUT", DEFAULT_TEST_TIMEOUT_S)

    # Resolve the pinned row BEFORE creating the temp dir / subprocess: a miss
    # is a GraderError (INFRA_GRADER) naming --prefetch, never a hub-id fallback.
    row = _pinned_row(spec, task)

    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"grade-{task.instance_id}-") as tmpdir:
        tmp = Path(tmpdir)
        preds = tmp / "preds.json"
        preds.write_text(
            json.dumps(
                [
                    {
                        "instance_id": task.instance_id,
                        "model_name_or_path": PREDICTION_TAG,
                        "model_patch": patch,
                    }
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        local = tmp / "dataset.jsonl"
        local.write_text(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        dataset_arg = str(local)

        run_tag = f"harness-{uuid.uuid4().hex[:12]}"
        cmd = _eval_command(spec, dataset_arg, preds, task.instance_id, run_tag, tmp, test_timeout)

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tmp),
                env=env,
                capture_output=True,
                text=True,
                timeout=grader_timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GraderError(
                f"cannot start the evaluation harness ({cmd[0]} -m {spec.eval_module}): {exc}. "
                f"Install it with: python3 -m pip install {spec.grader_distribution}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GraderError(
                f"{spec.eval_module} exceeded HARNESS_GRADER_TIMEOUT={grader_timeout}s "
                f"for {task.instance_id}"
            ) from exc

        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        instance_report, summary = _find_report(tmp, task.instance_id)

        if instance_report is not None:
            return _verdict_from_report(
                spec, task, instance_report, grader_version, proc.returncode, combined
            )

        # No per-instance report: the verdict is unknowable from the grader's
        # scored output.  Fold the instance's own logs into the evidence — the
        # build failure / patch-apply failure text lives there, not on stdout.
        evidence = combined + "\n" + _instance_log_text(tmp, task.instance_id)
        if summary is not None:
            return _verdict_from_summary(
                spec, task, summary, grader_version, proc.returncode, evidence
            )
        if _looks_like_sandbox_failure(evidence):
            return _sandbox_verdict(spec, task, grader_version, proc.returncode, evidence)
        raise GraderError(
            f"{spec.eval_module} produced no report for {task.instance_id} "
            f"(exit {proc.returncode}): {_clip(evidence[-400:])}"
        )


def _pinned_row(spec: SuiteSpec, task: Task) -> dict:
    """The dataset row for `task`, at the revision `load_tasks` pinned it to.

    Looks up `_ROW_CACHE` by the exact (dataset, split, revision) recorded in
    `task.metadata`; on a miss it reads the local HF cache under the same §5
    network gate `load_tasks` uses.  It NEVER falls back to the hub dataset id
    (which would let the grader resolve HEAD): if the pinned rows are not
    available it raises GraderError naming `--prefetch`, so the attempt is
    recorded as INFRA_GRADER rather than graded against an unknown revision.
    """
    split = str(task.metadata.get("split") or spec.split)
    if "revision" not in task.metadata:
        raise GraderError(
            f"{task.instance_id}: task carries no dataset revision (metadata.revision) — it "
            f"was not built by {spec.suite_name}'s load_tasks, so the pinned row cannot be "
            "located; rebuild the task list with load_tasks(seed_file)"
        )
    revision = task.metadata.get("revision")  # None == explicitly unpinned
    label = f"{spec.dataset}@{revision or UNPINNED_REVISION} (split={split})"
    rows = _ROW_CACHE.get((spec.dataset, split, revision))
    if rows is None:
        try:
            rows = _rows_for(spec, spec.dataset, split, revision)
        except AdapterConfigError as exc:
            raise GraderError(
                f"pinned dataset rows for {label} are not available on this host, so "
                f"{task.instance_id} cannot be graded against the pinned revision: {exc}. "
                + _prefetch_hint(spec)
            ) from exc
    row = rows.get(task.instance_id)
    if row is None:
        raise GraderError(
            f"{task.instance_id} is absent from {label}; the task list and the grader "
            "disagree about the dataset revision"
        )
    return _jsonable(row)


def _zero_counts(task: Task) -> tuple[dict, dict]:
    return (
        {"passed": 0, "total": len(task.fail_to_pass)},
        {"passed": 0, "total": len(task.pass_to_pass)},
    )


def _sandbox_verdict(
    spec: SuiteSpec, task: Task, grader_version: str, returncode: int, evidence: str, raw_extra: dict | None = None
) -> Verdict:
    f2p, p2p = _zero_counts(task)
    raw = {"returncode": returncode, "tail": evidence[-4000:]}
    if raw_extra:
        raw.update(raw_extra)
    return Verdict(
        resolved=False,
        error_code="INFRA_SANDBOX",
        detail=_clip(
            f"evaluation environment could not be prepared for {task.instance_id}: "
            + evidence[-400:]
        ),
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        grader=spec.grader,
        grader_version=grader_version,
        raw=raw,
    )


def _verdict_from_summary(
    spec: SuiteSpec,
    task: Task,
    summary: Mapping[str, Any],
    grader_version: str,
    returncode: int,
    evidence: str,
) -> Verdict:
    """Verdict when only upstream's run summary exists (no per-instance report).

    The summary carries membership lists but no test counts, so it can never
    produce a scored TESTS_FAIL / TESTS_REGRESSION / OK.  `error_ids` is where
    upstream puts *every* submitted instance that ended without a report —
    patch-apply failures and container-build failures alike — so the instance's
    logs decide between PATCH_MALFORMED (definitive upstream marker),
    INFRA_SANDBOX (docker/image markers) and GraderError (unknown).
    """
    iid = task.instance_id
    f2p, p2p = _zero_counts(task)
    raw = {
        "returncode": returncode,
        "summary": {k: v for k, v in summary.items() if k != "incomplete_ids"},
        "eval_module": spec.eval_module,
        "log_tail": evidence[-4000:],
    }

    if _in_summary_list(summary, "empty_patch_ids", iid):
        return Verdict(
            resolved=False,
            error_code="NO_PATCH",
            detail="evaluation harness saw no patch for this prediction",
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            grader=spec.grader,
            grader_version=grader_version,
            raw=raw,
        )

    completed = any(
        _in_summary_list(summary, k, iid) for k in ("completed_ids", "resolved_ids", "unresolved_ids")
    )
    if _in_summary_list(summary, "error_ids", iid) or not completed:
        if _looks_like_patch_apply_failure(evidence):
            return Verdict(
                resolved=False,
                error_code="PATCH_MALFORMED",
                detail=_clip(
                    f"patch did not apply to {task.repo}@{task.base_commit[:12]} "
                    "(git apply 3-way then patch -p1 both failed inside the evaluation container)"
                ),
                fail_to_pass=f2p,
                pass_to_pass=p2p,
                grader=spec.grader,
                grader_version=grader_version,
                raw=raw,
            )
        if _looks_like_sandbox_failure(evidence):
            return _sandbox_verdict(
                spec, task, grader_version, returncode, evidence, raw_extra={"summary": raw["summary"]}
            )
        where = "error_ids" if _in_summary_list(summary, "error_ids", iid) else "no id list"
        raise GraderError(
            f"{spec.eval_module} ended {iid} without a per-instance report (summary: {where}, "
            f"exit {returncode}) and the logs show neither a patch-apply failure nor a "
            f"sandbox failure: {_clip(evidence[-400:])}"
        )

    # Upstream lists it as completed, so a report.json existed when the summary
    # was written — yet none was readable now.  Counts are unknowable.
    raise GraderError(
        f"unparseable verdict for {iid}: the run summary lists it as completed "
        f"(resolved={_in_summary_list(summary, 'resolved_ids', iid)}) but no readable "
        "per-instance report.json with tests_status was found"
    )


def _verdict_from_report(
    spec: SuiteSpec,
    task: Task,
    report: Mapping[str, Any],
    grader_version: str,
    returncode: int,
    combined: str,
) -> Verdict:
    """Verdict from the per-instance report.json body.

    NO_PATCH / PATCH_MALFORMED are decided from the patch flags before the test
    counts are read; for anything else `tests_status` must be present and
    parseable or the verdict is unknowable (GraderError -> INFRA_GRADER).
    """
    iid = task.instance_id
    raw = {
        "returncode": returncode,
        "report": dict(report),
        "eval_module": spec.eval_module,
        "log_tail": combined[-4000:],
    }

    applied = report.get("patch_successfully_applied")
    exists = report.get("patch_exists", True)
    if report.get("patch_is_None") or exists is False:
        f2p, p2p = _zero_counts(task)
        return Verdict(
            resolved=False,
            error_code="NO_PATCH",
            detail="evaluation harness saw no patch for this prediction",
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            grader=spec.grader,
            grader_version=grader_version,
            raw=raw,
        )
    if applied is False:
        f2p, p2p = _zero_counts(task)
        return Verdict(
            resolved=False,
            error_code="PATCH_MALFORMED",
            detail=_clip(
                f"patch did not apply to {task.repo}@{task.base_commit[:12]} "
                "(git apply 3-way then patch -p1 both failed inside the evaluation container)"
            ),
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            grader=spec.grader,
            grader_version=grader_version,
            raw=raw,
        )

    tests_status = report.get("tests_status")
    if not isinstance(tests_status, dict):
        raise GraderError(
            f"unparseable verdict for {iid}: report has no tests_status "
            f"(keys: {sorted(str(k) for k in report.keys())[:12]}); the evaluation harness "
            "report format may have changed"
        )
    f2p = _counts(tests_status.get("FAIL_TO_PASS"), "FAIL_TO_PASS", iid, len(task.fail_to_pass))
    p2p = _counts(tests_status.get("PASS_TO_PASS"), "PASS_TO_PASS", iid, len(task.pass_to_pass))
    resolved = bool(report.get("resolved"))

    if resolved:
        return Verdict(
            resolved=True,
            error_code="OK",
            detail=(
                f"fail_to_pass {f2p['passed']}/{f2p['total']}, "
                f"pass_to_pass {p2p['passed']}/{p2p['total']}"
            ),
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            grader=spec.grader,
            grader_version=grader_version,
            raw=raw,
        )

    f2p_all_pass = f2p["total"] > 0 and f2p["passed"] == f2p["total"]
    p2p_broken = p2p["passed"] < p2p["total"]
    if f2p_all_pass and p2p_broken:
        code = "TESTS_REGRESSION"
        detail = (
            f"all {f2p['total']} fail_to_pass tests pass but "
            f"{p2p['total'] - p2p['passed']}/{p2p['total']} pass_to_pass tests regressed"
        )
    else:
        code = "TESTS_FAIL"
        detail = (
            f"fail_to_pass {f2p['passed']}/{f2p['total']} after patch; "
            f"pass_to_pass {p2p['passed']}/{p2p['total']}"
        )
    return Verdict(
        resolved=False,
        error_code=code,
        detail=_clip(detail),
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        grader=spec.grader,
        grader_version=grader_version,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# environment digest
# --------------------------------------------------------------------------- #


def _digest_seed_file(spec: SuiteSpec, seed_file: Path | str | None) -> Path:
    """Seed file whose `source.revision` the digest records.

    `environment_digest()` takes no arguments by contract (§5), so the seed file
    is resolved from: an explicit argument (operator CLI) -> `HARNESS_SEED_FILE`
    (`HARNESS_SEED_FILE_<SUITE>` first) -> the suite's default seed file.  Every
    call site in one run (manifest build, preflight, agent) must resolve the
    same file, so a `--seed-file` override should be exported as
    HARNESS_SEED_FILE for the whole run.
    """
    if seed_file is not None:
        return Path(seed_file)
    override = spec.env("SEED_FILE")
    if override:
        return Path(override)
    return _REPO_ROOT / spec.default_seed_file


def _seed_revision_marker(spec: SuiteSpec, seed_file: Path | str | None) -> str:
    """The seed file's pinned `source.revision`, or `UNPINNED_REVISION`.

    Deliberately lenient about the pin (unlike `load_seed`): a placeholder
    revision yields the explicit "unpinned" marker rather than raising, so the
    digest is computable — and different from any pinned run's — on the
    HARNESS_ALLOW_UNPINNED_DATASET=1 path.  A missing or malformed seed file is
    still a config error.
    """
    path = _digest_seed_file(spec, seed_file)
    doc = _read_json(path, "seed file")
    if not isinstance(doc, dict):
        raise AdapterConfigError(f"seed file {path}: top level must be an object")
    source = doc.get("source")
    revision = source.get("revision") if isinstance(source, dict) else None
    if isinstance(revision, str) and not _looks_unresolved(revision):
        return revision
    return UNPINNED_REVISION


def environment_digest(spec: SuiteSpec, seed_file: Path | str | None = None) -> str:
    """`sha256:<hex>` identifying the grading environment.

    Covers everything that could change a verdict for a fixed (task, patch):
    the adapter itself, the pinned dataset *at its pinned revision* (or the
    explicit "unpinned" marker), the evaluation harness artifact, the container
    runtime, and the grading knobs read from the environment.  The docker image
    cache is deliberately not covered: `--cache_level env` reuses environment
    images whose recipe is fixed by the swebench distribution hashed here.
    """
    dataset_revision = _seed_revision_marker(spec, seed_file)
    adapter_file = _REPO_ROOT / "harness" / "adapters" / f"{spec.suite_name.replace('-', '_')}.py"
    try:
        adapter_sha = _sha256_hex(adapter_file.read_bytes())
    except OSError:
        adapter_sha = "unavailable"
    try:
        base_sha = _sha256_hex(Path(__file__).read_bytes())
    except OSError:
        base_sha = "unavailable"

    docker_version = "unavailable"
    docker_bin = shutil.which("docker")
    if docker_bin:
        try:
            out = subprocess.run(
                [docker_bin, "version", "--format", "{{.Server.Version}}"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                docker_version = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    payload = {
        "schema": "grading-environment/v2",
        "suite": spec.suite_name,
        "adapter_version": spec.adapter_version,
        "adapter_sha256": adapter_sha,
        "adapter_base_sha256": base_sha,
        "dataset": spec.dataset,
        "dataset_revision": dataset_revision,
        "split": spec.split,
        "grader": spec.grader,
        "grader_distribution": spec.grader_distribution,
        "grader_version": _dist_version(spec.grader_distribution) or "unavailable",
        "grader_record_sha256": _dist_record_sha256(spec.grader_distribution) or "unavailable",
        "eval_module": spec.eval_module,
        "eval_cmd_override": spec.env("EVAL_CMD") or "",
        "image_template": spec.env("IMAGE_TEMPLATE") or spec.image_template or "",
        "container_runtime": f"docker/{docker_version}",
        "test_timeout_s": _int_env("HARNESS_TEST_TIMEOUT", DEFAULT_TEST_TIMEOUT_S),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "datasets_version": _dist_version("datasets") or "unavailable",
    }
    return "sha256:" + _sha256_hex(_canonical_bytes(payload))


# --------------------------------------------------------------------------- #
# operator CLI (prefetch / inspect) — never used by the harness itself
# --------------------------------------------------------------------------- #


def main(spec: SuiteSpec, argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog=f"python3 -m harness.adapters.{spec.suite_name.replace('-', '_')}",
        description=f"{spec.suite_name} adapter: inspect the seeded task set or prefetch rows.",
    )
    parser.add_argument("--seed-file", default=str(_REPO_ROOT / spec.default_seed_file))
    parser.add_argument("--partitions", default=None)
    parser.add_argument("--prefetch", action="store_true", help="download the pinned dataset (needs HARNESS_ALLOW_NETWORK=1)")
    parser.add_argument("--list", action="store_true", help="print seeded instance ids, one per line")
    parser.add_argument("--digest", action="store_true", help="print environment_digest()")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.digest:
        print(environment_digest(spec, seed_file=Path(args.seed_file)))
        return 0

    seed = load_seed(spec, Path(args.seed_file))
    if args.list:
        for instance_id in seed.instance_ids:
            print(instance_id)
        return 0

    if args.prefetch and not _network_allowed():
        print(
            "refusing to download: set HARNESS_ALLOW_NETWORK=1 to permit network access",
            file=sys.stderr,
        )
        return 2

    tasks = load_tasks(spec, Path(args.seed_file), Path(args.partitions) if args.partitions else None)
    print(
        f"{spec.suite_name}: {len(tasks)} tasks from {seed.dataset}@"
        f"{seed.revision or 'unpinned'} (split={seed.split}), seed file sha256 {seed.sha256}",
        file=sys.stderr,
    )
    print(json.dumps({"suite": spec.suite_name, "tasks": len(tasks), "seed_sha256": seed.sha256}))
    return 0
