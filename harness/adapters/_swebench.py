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
   `grade()` for the exact invocation and the report parsing.
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


def _checked_revision(seed_file: Path, revision: str | None) -> str | None:
    """Reject a seed file whose dataset revision is still a placeholder.

    An unpinned dataset silently changes what "the same 100 instances" means, so
    this is a hard configuration error rather than a warning. Set
    HARNESS_ALLOW_UNPINNED_DATASET=1 to proceed against the dataset head while
    the pin is being resolved; the run is not publication-grade.
    """
    if revision is None:
        looks_unresolved = True
    else:
        lowered = revision.lower()
        looks_unresolved = any(m in lowered for m in _UNRESOLVED_REVISION_MARKERS)
    if not looks_unresolved:
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


def _load_rows(spec: SuiteSpec, seed: Seed) -> dict[str, dict]:
    """Return {instance_id: row} for the seeded ids only, pinned to the seed revision."""
    key = (seed.dataset, seed.split, seed.revision)
    cached = _ROW_CACHE.get(key)
    if cached is None:
        offline = not _network_allowed()
        datasets = _import_datasets(offline)
        kwargs: dict[str, Any] = {"split": seed.split}
        if seed.revision:
            kwargs["revision"] = seed.revision
        try:
            ds = datasets.load_dataset(seed.dataset, **kwargs)
        except Exception as exc:  # datasets raises a wide variety of types
            hint = (
                "No local HuggingFace cache entry and network access is disabled. "
                "Prefetch once on this host with\n"
                f"    HARNESS_ALLOW_NETWORK=1 python3 -m harness.adapters."
                f"{spec.suite_name.replace('-', '_')} --prefetch\n"
                "or export HARNESS_ALLOW_NETWORK=1 for this run."
                if offline
                else "Network access was permitted but the download failed."
            )
            raise DatasetUnavailable(
                f"could not load {seed.dataset} (split={seed.split}, "
                f"revision={seed.revision or 'unpinned'}): {exc}. {hint}"
            ) from exc
        cached = {}
        for row in ds:
            instance_id = row.get("instance_id")
            if isinstance(instance_id, str):
                cached[instance_id] = dict(row)
        _ROW_CACHE[key] = cached

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
    metadata = {
        "dataset": spec.dataset,
        "split": spec.split,
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
    return [_build_task(spec, rows[i], i, partitions) for i in seed.instance_ids]


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
#   dataset.jsonl  the single pinned dataset row, so the grader evaluates exactly
#                  the revision the seed file pins (`--dataset_name` accepts a
#                  local .jsonl path).  Set HARNESS_DATASET_MODE=hub to pass the
#                  hub dataset id instead.
#   preds.json     [{"instance_id", "model_name_or_path": PREDICTION_TAG,
#                    "model_patch": <patch>}]
#
# The subprocess runs with cwd=<temp dir>, so the harness's `logs/` tree and its
# summary report land inside the temp dir and are removed with it.
#
# Report parsing (in order): logs/run_evaluation/<run_id>/<tag>/<id>/report.json,
# any nested report.json, then the top-level `<tag>.<run_id>.json` summary.
#
# Verdict mapping (CONTRACTS §4):
#   patch empty/whitespace                -> NO_PATCH   (no environment built)
#   patch did not apply                   -> PATCH_MALFORMED
#   resolved                              -> OK, resolved=True
#   all FAIL_TO_PASS pass, PASS_TO_PASS regressed -> TESTS_REGRESSION
#   otherwise                             -> TESTS_FAIL
#   docker/image/setup failure            -> INFRA_SANDBOX (returned, not raised)
#   grader crash / timeout / no report    -> GraderError -> INFRA_GRADER
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
        "--cache_level",
        "none",
        "--clean",
        "False",
        "--timeout",
        str(test_timeout),
    ]


def _find_report(tmp: Path, instance_id: str) -> dict | None:
    candidates: list[Path] = []
    nested = sorted(tmp.rglob("report.json"))
    candidates.extend(nested)
    candidates.extend(sorted(p for p in tmp.glob("*.json") if p.name not in {"preds.json"}))
    for path in candidates:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        if instance_id in doc and isinstance(doc[instance_id], dict):
            return doc[instance_id]
        if "tests_status" in doc or "patch_successfully_applied" in doc:
            return doc
        if "resolved_ids" in doc:  # summary report: usable, but counts-free
            return {
                "_summary_only": True,
                "resolved": instance_id in (doc.get("resolved_ids") or []),
                "patch_exists": instance_id not in (doc.get("empty_patch_instances") or []),
                "patch_successfully_applied": instance_id
                not in (doc.get("error_instances") or []),
            }
    return None


def _counts(block: Any, expected_total: int) -> dict:
    if isinstance(block, dict):
        success = block.get("success") or []
        failure = block.get("failure") or []
        if isinstance(success, list) and isinstance(failure, list):
            total = len(success) + len(failure)
            return {"passed": len(success), "total": total or expected_total}
    return {"passed": 0, "total": expected_total}


def _looks_like_sandbox_failure(output: str) -> bool:
    lowered = output.lower()
    return any(marker in lowered for marker in _SANDBOX_MARKERS)


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
    dataset_mode = (spec.env("DATASET_MODE", "local") or "local").lower()

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

        if dataset_mode == "local":
            row = _pinned_row(spec, task)
            if row is None:
                dataset_arg = spec.dataset
            else:
                local = tmp / "dataset.jsonl"
                local.write_text(
                    json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                dataset_arg = str(local)
        else:
            dataset_arg = spec.dataset

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
        report = _find_report(tmp, task.instance_id)

        if report is None:
            if _looks_like_sandbox_failure(combined):
                return Verdict(
                    resolved=False,
                    error_code="INFRA_SANDBOX",
                    detail=_clip(
                        f"evaluation environment could not be prepared for {task.instance_id}: "
                        + combined[-400:]
                    ),
                    fail_to_pass={"passed": 0, "total": len(task.fail_to_pass)},
                    pass_to_pass={"passed": 0, "total": len(task.pass_to_pass)},
                    grader=spec.grader,
                    grader_version=grader_version,
                    raw={"returncode": proc.returncode, "tail": combined[-4000:]},
                )
            raise GraderError(
                f"{spec.eval_module} produced no report for {task.instance_id} "
                f"(exit {proc.returncode}): {_clip(combined[-400:])}"
            )

        return _verdict_from_report(spec, task, report, grader_version, proc.returncode, combined)


def _pinned_row(spec: SuiteSpec, task: Task) -> dict | None:
    """The dataset row for `task`, from the cache populated by load_tasks.

    Returns None when the row is not cached (e.g. grade() called in isolation);
    the caller then falls back to the hub dataset id.
    """
    key = (spec.dataset, str(task.metadata.get("split") or spec.split), None)
    for (dataset, split, _rev), rows in _ROW_CACHE.items():
        if dataset != spec.dataset:
            continue
        if split != key[1]:
            continue
        row = rows.get(task.instance_id)
        if row is not None:
            return _jsonable(row)
    return None


def _verdict_from_report(
    spec: SuiteSpec,
    task: Task,
    report: Mapping[str, Any],
    grader_version: str,
    returncode: int,
    combined: str,
) -> Verdict:
    tests_status = report.get("tests_status") or {}
    f2p = _counts(tests_status.get("FAIL_TO_PASS"), len(task.fail_to_pass))
    p2p = _counts(tests_status.get("PASS_TO_PASS"), len(task.pass_to_pass))
    resolved = bool(report.get("resolved"))
    raw = {
        "returncode": returncode,
        "report": dict(report),
        "eval_module": spec.eval_module,
        "log_tail": combined[-4000:],
    }

    applied = report.get("patch_successfully_applied")
    exists = report.get("patch_exists", True)
    if report.get("patch_is_None") or exists is False:
        return Verdict(
            resolved=False,
            error_code="NO_PATCH",
            detail="evaluation harness saw no patch for this prediction",
            fail_to_pass={"passed": 0, "total": len(task.fail_to_pass)},
            pass_to_pass={"passed": 0, "total": len(task.pass_to_pass)},
            grader=spec.grader,
            grader_version=grader_version,
            raw=raw,
        )
    if applied is False:
        return Verdict(
            resolved=False,
            error_code="PATCH_MALFORMED",
            detail=_clip(
                f"patch did not apply to {task.repo}@{task.base_commit[:12]} "
                "(git apply 3-way then patch -p1 both failed inside the evaluation container)"
            ),
            fail_to_pass={"passed": 0, "total": len(task.fail_to_pass)},
            pass_to_pass={"passed": 0, "total": len(task.pass_to_pass)},
            grader=spec.grader,
            grader_version=grader_version,
            raw=raw,
        )

    if resolved:
        if report.get("_summary_only"):
            f2p = {"passed": f2p["total"], "total": f2p["total"]}
            p2p = {"passed": p2p["total"], "total": p2p["total"]}
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


def environment_digest(spec: SuiteSpec) -> str:
    """`sha256:<hex>` identifying the grading environment.

    Covers everything that could change a verdict for a fixed (task, patch):
    the adapter itself, the pinned dataset, the evaluation harness artifact, the
    container runtime, and the grading knobs read from the environment.
    """
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
        "schema": "grading-environment/v1",
        "suite": spec.suite_name,
        "adapter_version": spec.adapter_version,
        "adapter_sha256": adapter_sha,
        "adapter_base_sha256": base_sha,
        "dataset": spec.dataset,
        "split": spec.split,
        "grader": spec.grader,
        "grader_distribution": spec.grader_distribution,
        "grader_version": _dist_version(spec.grader_distribution) or "unavailable",
        "grader_record_sha256": _dist_record_sha256(spec.grader_distribution) or "unavailable",
        "eval_module": spec.eval_module,
        "eval_cmd_override": spec.env("EVAL_CMD") or "",
        "dataset_mode": (spec.env("DATASET_MODE", "local") or "local").lower(),
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
        print(environment_digest(spec))
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
