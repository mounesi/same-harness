#!/usr/bin/env python3
"""agenttask.py — suite adapter for the 50 internal AgentTask-derived coding tasks.

Implements the CONTRACTS.md §5 adapter interface. Each task is a repo snapshot + an
issue text + a set of hidden tests, graded exactly like SWE-bench: apply the agent's
diff to the snapshot, overlay the hidden tests, run them, and require every
`fail_to_pass` test to go green while every `pass_to_pass` test stays green.

    CONSENT (AI-2955) — READ THIS BEFORE TOUCHING THIS FILE
    CONSENT_CLASS is "restricted". Task text, repo snapshots, hidden tests, patches and
    trajectories for this suite are customer-derived and MUST NOT enter git history,
    the paper, or any public artifact until the consent decision lands. This module
    therefore reads task *content* from an out-of-tree data pack and refuses to load one
    that lives inside the repo working tree. Only ids (suites/agenttask/seed.json) and
    aggregate counts are git-safe. See suites/agenttask/README.md.

Data pack layout (out of tree, resolved by _data_dir()):

    <pack>/PACK.json                      {"schema":"agenttask-pack/v1", ...}
    <pack>/tasks/<instance_id>.json       one agenttask-task/v1 record per task
    <pack>/snapshots/<instance_id>.tar.gz agent-visible repo snapshot (no hidden tests)
    <pack>/hidden-tests/<instance_id>.tar.gz  grader-only overlay, applied AFTER the patch

The record format is documented in suites/agenttask/README.md, which is the normative
description of the pack; this module is its reference reader.

Grading timeouts (CONTRACTS.md §4):

    `environment.test_timeout_s` is a PER-ATTEMPT budget: it bounds the whole grading of one
    attempt — every fail_to_pass and pass_to_pass node id together — for BOTH runners. The
    pytest runner spends it on its single batched invocation; the exit-code runner runs one
    process per node id and charges each against the same shared deadline. Hitting it is a
    grader hang: grade() raises GraderError and the caller records INFRA_GRADER (excluded
    from the denominator) rather than a scored TESTS_FAIL, because a verdict was never
    observed. The scope actually applied is recorded in every verdict's raw dict as
    `timeout_scope`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

# The adapter is imported both as `harness.adapters.agenttask` (via the registry) and
# occasionally as a bare script during development; make the repo root importable so the
# shared harness modules resolve either way.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.types import GraderError, Prompt, Task, Verdict  # noqa: E402

# `harness.prompts` is imported lazily, inside the two functions that need it, so a broken
# prompt directory surfaces as a prompt error at render time rather than as an unimportable
# adapter — the same shape the SWE-bench adapters use.

# --- module contract (CONTRACTS.md §5) --------------------------------------------------

SUITE_NAME = "agenttask"
ADAPTER_VERSION = "1.0.0"
CONSENT_CLASS = "restricted"

GRADER = "agenttask-eval"
GRADER_VERSION = "1.0.0"

SEED_SCHEMA = "suite-seed/v1"
TASK_SCHEMA = "agenttask-task/v1"
PACK_SCHEMA = "agenttask-pack/v1"

DEFAULT_SEED_FILE = _REPO_ROOT / "suites" / "agenttask" / "seed.json"

INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Grading defaults. These are *task environment* properties, not harness properties: the
# held-constant knobs (iteration budget, sampling, prompt) live in harness/agent_config.json.
# DEFAULT_TEST_TIMEOUT_S is the budget for grading ONE ATTEMPT (all node ids, both runners,
# see the module docstring) — not per test and not per process. DEFAULT_SETUP_TIMEOUT_S is
# per setup command.
DEFAULT_TEST_CMD = "python -m pytest -rA -p no:cacheprovider {tests}"
DEFAULT_TEST_TIMEOUT_S = 900
DEFAULT_SETUP_TIMEOUT_S = 1800
TIMEOUT_SCOPE = "per_attempt_batch"  # recorded in Verdict.raw["timeout_scope"]
DETAIL_MAX = 512
OUTPUT_TAIL_CHARS = 2000


class AgentTaskDataError(RuntimeError):
    """The task pack is missing, malformed, or in a place consent policy forbids.

    Raised from load_tasks(); run.sh maps a load-time failure to exit 2 (config).
    """


# --- small helpers ----------------------------------------------------------------------



def _is_within(child: Path, parent: Path) -> bool:
    """Path.is_relative_to equivalent for CPython < 3.9.

    The grading preflight calls environment_digest() on every host, including boxes with
    an older default python3, so this must not depend on a 3.9+ method.
    """
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False

def _canonical_json(obj: Any) -> str:
    """Canonical JSON per CONTRACTS.md §0 (compact form, used only for hashing)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_digest(root: Path) -> str:
    """Directory digest, CONTRACTS.md §2.4. Equivalent to `sha256sum | LC_ALL=C sort -k2`."""
    skip_dirs = {".git", ".cache", "__pycache__"}
    pairs: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = set(rel.split("/")[:-1])
        if parts & skip_dirs:
            continue
        if path.name in (".DS_Store",) or path.suffix == ".pyc":
            continue
        pairs.append((rel, _sha256_file(path)))
    pairs.sort(key=lambda p: p[0].encode("utf-8"))
    stream = "".join(f"{h}  {rel}\n" for rel, h in pairs)
    return _sha256_bytes(stream.encode("utf-8"))


def _truncate(text: str, limit: int = DETAIL_MAX) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _template_id() -> str:
    """The one template id every suite shares (CONTRACTS.md §5.2)."""
    try:
        from harness import prompts as harness_prompts

        tid = getattr(harness_prompts, "TEMPLATE_ID", None)
        if tid:
            return str(tid)
    except ImportError:
        pass
    marker = _REPO_ROOT / "harness" / "prompts" / "TEMPLATE_ID"
    try:
        return marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AgentTaskDataError(
            f"prompt template id not readable at {marker}: {exc}; harness/prompts/ is required"
        ) from exc


# --- data pack resolution ---------------------------------------------------------------


def _data_dir() -> Path:
    """Locate the out-of-tree AgentTask data pack.

    Order: $AGENTTASK_DATA_DIR, /persistent/agenttask, ~/.harness/agenttask.
    A pack inside the repo working tree is refused outright — that is the consent guard.
    """
    candidates: list[Path] = []
    env = os.environ.get("AGENTTASK_DATA_DIR", "").strip()
    if env:
        candidates.append(Path(env))
    else:
        candidates.append(Path("/persistent/agenttask"))
        candidates.append(Path.home() / ".harness" / "agenttask")

    for cand in candidates:
        resolved = cand.expanduser().resolve()
        if _is_within(resolved, _REPO_ROOT):
            raise AgentTaskDataError(
                f"AgentTask data pack at {resolved} is inside the repo working tree; "
                "customer-derived task content must never be reachable from git (AI-2955)"
            )
        if (resolved / "tasks").is_dir():
            return resolved

    tried = ", ".join(str(c) for c in candidates)
    raise AgentTaskDataError(
        f"no AgentTask data pack found (looked in: {tried}); set AGENTTASK_DATA_DIR"
    )


def _pack_info(data_dir: Path) -> dict:
    meta = data_dir / "PACK.json"
    if not meta.is_file():
        return {"schema": PACK_SCHEMA, "pack_revision": "unresolved", "pack_sha256": None}
    info = json.loads(meta.read_text(encoding="utf-8"))
    if info.get("schema") != PACK_SCHEMA:
        raise AgentTaskDataError(f"{meta}: schema is not {PACK_SCHEMA}")
    return info


# --- seed file --------------------------------------------------------------------------


def _load_seed(seed_file: Path) -> list[str]:
    if not seed_file.is_file():
        raise AgentTaskDataError(f"seed file not found: {seed_file}")
    doc = json.loads(seed_file.read_text(encoding="utf-8"))

    if doc.get("schema") != SEED_SCHEMA:
        raise AgentTaskDataError(f"{seed_file}: schema is not {SEED_SCHEMA}")
    if doc.get("suite") != SUITE_NAME:
        raise AgentTaskDataError(f"{seed_file}: suite is {doc.get('suite')!r}, expected {SUITE_NAME!r}")
    if doc.get("placeholder") and os.environ.get("AGENTTASK_ALLOW_PLACEHOLDER") != "1":
        raise AgentTaskDataError(
            f"{seed_file} is a PLACEHOLDER seed file — regenerate it with "
            "suites/generate_seeds.py before running the suite "
            "(set AGENTTASK_ALLOW_PLACEHOLDER=1 only for harness self-tests)"
        )

    ids = list(doc.get("instance_ids") or [])
    if not ids:
        raise AgentTaskDataError(f"{seed_file}: instance_ids is empty")
    if len(set(ids)) != len(ids):
        raise AgentTaskDataError(f"{seed_file}: instance_ids contains duplicates")
    if doc.get("count") != len(ids):
        raise AgentTaskDataError(f"{seed_file}: count {doc.get('count')} != len(instance_ids) {len(ids)}")
    for iid in ids:
        if not INSTANCE_ID_RE.match(iid):
            raise AgentTaskDataError(f"{seed_file}: instance_id {iid!r} is not ^[A-Za-z0-9._-]+$")

    want = doc.get("instance_ids_sha256")
    got = _sha256_bytes(("\n".join(sorted(ids)) + "\n").encode("utf-8"))
    if want and want != got:
        raise AgentTaskDataError(f"{seed_file}: instance_ids_sha256 mismatch ({want} != {got})")
    return ids


def _default_partitions_path() -> Path | None:
    env = os.environ.get("HARNESS_PARTITIONS", "").strip()
    if env:
        return Path(env)
    default = _REPO_ROOT / "suites" / "partitions.json"
    return default if default.is_file() else None


def _partition_map(partitions: Any) -> dict[str, str]:
    """qualified_id -> partition name. An absent file means every task is "unpartitioned"."""
    if partitions is None:
        return {}
    if isinstance(partitions, Mapping):
        return dict(partitions)
    path = Path(partitions)
    if not path.is_file():
        raise AgentTaskDataError(f"partitions file not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for name, block in (doc.get("partitions") or {}).items():
        for qid in block.get("ids") or []:
            out[qid] = name
    return out


# --- adapter API ------------------------------------------------------------------------


def load_tasks(
    seed_file: Path | str = DEFAULT_SEED_FILE,
    partitions_file: Path | None = None,
) -> list[Task]:
    """Build the Task list for this suite, in seed-file order. Never touches the network.

    `partitions_file` is the same optional extension over the §5 signature that the
    SWE-bench adapters take: when omitted it falls back to `$HARNESS_PARTITIONS`, then to
    `suites/partitions.json`, then to `partition="unpartitioned"` (CONTRACTS.md §3.1).
    """
    seed_file = Path(seed_file)
    ids = _load_seed(seed_file)
    data_dir = _data_dir()
    pack = _pack_info(data_dir)
    part = _partition_map(
        partitions_file if partitions_file is not None else _default_partitions_path()
    )

    missing = [i for i in ids if not (data_dir / "tasks" / f"{i}.json").is_file()]
    if missing:
        raise AgentTaskDataError(
            f"{len(missing)} of {len(ids)} task records missing from {data_dir}/tasks "
            f"(first: {missing[0]})"
        )

    tasks: list[Task] = []
    for iid in ids:
        record = json.loads((data_dir / "tasks" / f"{iid}.json").read_text(encoding="utf-8"))
        tasks.append(_build_task(iid, record, data_dir, pack, part))
    return tasks


def _build_task(
    iid: str, record: dict, data_dir: Path, pack: dict, part: Mapping[str, str]
) -> Task:
    if record.get("schema") != TASK_SCHEMA:
        raise AgentTaskDataError(f"task {iid}: schema is not {TASK_SCHEMA}")
    if record.get("instance_id") != iid:
        raise AgentTaskDataError(f"task {iid}: instance_id in record is {record.get('instance_id')!r}")

    problem = record.get("problem_statement")
    if not isinstance(problem, str) or not problem.strip():
        raise AgentTaskDataError(f"task {iid}: problem_statement is empty")

    f2p = tuple(record.get("fail_to_pass") or ())
    p2p = tuple(record.get("pass_to_pass") or ())
    if not f2p:
        raise AgentTaskDataError(f"task {iid}: fail_to_pass is empty — a task with no red test is not gradable")
    # Node ids must be non-empty strings. Spaces are legal (pytest parametrize ids such as
    # `test_parse[2020-01-01 00:00:00-utc]`) and are matched verbatim against the -rA summary.
    for field, ids in (("fail_to_pass", f2p), ("pass_to_pass", p2p)):
        for nid in ids:
            if not isinstance(nid, str) or not nid.strip():
                raise AgentTaskDataError(f"task {iid}: {field} contains a non-string or empty node id: {nid!r}")

    snapshot = dict(record.get("snapshot") or {})
    hidden = dict(record.get("hidden_tests") or {})
    if not snapshot.get("path"):
        raise AgentTaskDataError(f"task {iid}: snapshot.path is missing")
    if not hidden.get("path"):
        raise AgentTaskDataError(f"task {iid}: hidden_tests.path is missing")

    env_in = dict(record.get("environment") or {})
    environment = {
        "image": env_in.get("image", ""),
        "setup_cmds": list(env_in.get("setup_cmds") or []),
        "test_cmd": env_in.get("test_cmd") or DEFAULT_TEST_CMD,
        "runner": env_in.get("runner", "pytest"),
        "report_json": env_in.get("report_json"),
        "test_timeout_s": int(env_in.get("test_timeout_s", DEFAULT_TEST_TIMEOUT_S)),
        "setup_timeout_s": int(env_in.get("setup_timeout_s", DEFAULT_SETUP_TIMEOUT_S)),
    }
    if environment["runner"] not in ("pytest", "exit-code"):
        raise AgentTaskDataError(f"task {iid}: unknown runner {environment['runner']!r}")

    qualified_id = f"{SUITE_NAME}::{iid}"
    # metadata is adapter-private and never read by the harness (CONTRACTS.md §5.1). It
    # deliberately carries no task text — only the pointers grade() needs.
    metadata = {
        "consent_class": CONSENT_CLASS,
        "data_dir": str(data_dir),
        "pack_revision": pack.get("pack_revision", "unresolved"),
        "snapshot": snapshot,
        "hidden_tests": hidden,
        "labels": list(record.get("labels") or []),
        "origin": record.get("origin", "internal/agenttask"),
    }

    return Task(
        suite=SUITE_NAME,
        instance_id=iid,
        qualified_id=qualified_id,
        repo=record.get("repo", "") or "",
        base_commit=snapshot.get("base_commit", "") or "",
        problem_statement=problem,
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        environment=environment,
        partition=part.get(qualified_id, "unpartitioned"),
        metadata=metadata,
        source_sha256=_sha256_bytes(_canonical_json(record).encode("utf-8")),
    )


def build_prompt(task: Task) -> Prompt:
    """Render the ONE shared template. The harness is the control variable: this adapter
    supplies different values, never a different template or a suite-specific preamble."""
    template_id = _template_id()
    from harness import prompts as harness_prompts

    prompt = harness_prompts.render(
        template_id,
        {
            "problem_statement": task.problem_statement,
            "repo": task.repo,
            "test_cmd": task.environment.get("test_cmd", DEFAULT_TEST_CMD),
        },
    )
    if prompt.template_id != template_id:
        raise AgentTaskDataError(
            f"prompt template drift: rendered {prompt.template_id!r}, expected {template_id!r}"
        )
    return prompt


def grade(task: Task, patch: str) -> Verdict:
    """Apply `patch` to a throwaway copy of the snapshot, overlay the hidden tests, run them.

    Returns (never raises) for task-level failures. Raises GraderError only when the grading
    machinery itself is broken; the caller maps that to INFRA_GRADER.
    """
    if not patch or not patch.strip():
        return _verdict(False, "NO_PATCH", "agent produced no diff", {}, task, ran=False)

    base = os.environ.get("AGENTTASK_WORKDIR") or None
    try:
        with tempfile.TemporaryDirectory(prefix="agenttask-grade-", dir=base) as tmp:
            return _grade_in(Path(tmp), task, patch)
    except GraderError:
        raise
    except OSError as exc:  # temp dir / disk problems are grader infrastructure
        raise GraderError(f"agenttask grader io error: {exc}") from exc


def _grade_in(tmp: Path, task: Task, patch: str) -> Verdict:
    work = tmp / "work"
    env = task.environment

    try:
        _materialize_snapshot(task, work)
    except AgentTaskDataError as exc:
        return _verdict(False, "INFRA_SANDBOX", str(exc), {}, task, ran=False)

    applied, how = _apply_patch(work, patch, tmp)
    if not applied:
        return _verdict(False, "PATCH_MALFORMED", f"patch did not apply ({how})", {}, task, ran=False)

    try:
        overwritten = _overlay_hidden_tests(task, work)
    except AgentTaskDataError as exc:
        return _verdict(False, "INFRA_SANDBOX", str(exc), {}, task, ran=False)

    for cmd in env["setup_cmds"]:
        rc, out, setup_timed_out = _run(cmd, work, env["setup_timeout_s"])
        if rc != 0 or setup_timed_out:
            why = (
                f"setup command exceeded setup_timeout_s={env['setup_timeout_s']}"
                if setup_timed_out
                else f"setup command failed (rc={rc})"
            )
            return _verdict(
                False,
                "INFRA_SANDBOX",
                why,
                {
                    "setup_cmd_rc": rc,
                    "setup_timed_out": setup_timed_out,
                    "output_tail": out[-OUTPUT_TAIL_CHARS:],
                },
                task,
                ran=False,
            )

    node_ids = list(task.fail_to_pass) + list(task.pass_to_pass)
    test_timeout_s = int(env["test_timeout_s"])
    statuses, rc, out, timed_out = _run_tests(work, env, node_ids)

    if timed_out:
        # A grader hang, not a wrong answer: no verdict was observed, so this is INFRA_GRADER
        # (excluded from the denominator, CONTRACTS.md §4) rather than a scored TESTS_FAIL.
        raise GraderError(
            f"agenttask grader: test run exceeded {test_timeout_s}s "
            f"(scope={TIMEOUT_SCOPE}, runner={env.get('runner')}, "
            f"{len(node_ids)} node ids); output tail: {_truncate(out[-OUTPUT_TAIL_CHARS:], 300)}"
        )

    # A runner that never ran is not a verdict. rc 127 (command not found), a missing pytest
    # module, or output with no pytest session summary at all means NOTHING was observed —
    # scoring it would book a host-configuration problem as 0/N against the model.
    runner = str(env.get("runner") or "pytest")
    if rc == 127 or "No module named pytest" in out or (
        runner == "pytest" and not statuses and "passed" not in out and "failed" not in out
        and "error" not in out.lower()
    ):
        raise GraderError(
            f"agenttask grader: the {runner} runner did not run (rc={rc}); "
            f"output tail: {_truncate(out[-OUTPUT_TAIL_CHARS:], 300)}"
        )

    f2p_passed = sum(1 for t in task.fail_to_pass if statuses.get(t) == "passed")
    p2p_passed = sum(1 for t in task.pass_to_pass if statuses.get(t) == "passed")
    f2p = {"passed": f2p_passed, "total": len(task.fail_to_pass)}
    p2p = {"passed": p2p_passed, "total": len(task.pass_to_pass)}

    raw = {
        "returncode": rc,
        "timed_out": timed_out,
        "test_timeout_s": test_timeout_s,
        "timeout_scope": TIMEOUT_SCOPE,
        "statuses": statuses,
        "hidden_tests_overwrote_patched_paths": overwritten,
        "patch_applied_with": how,
        "output_tail": out[-OUTPUT_TAIL_CHARS:],
    }

    if f2p_passed == f2p["total"] and p2p_passed == p2p["total"]:
        return _verdict(True, "OK", "all fail_to_pass green, no regressions", raw, task, f2p=f2p, p2p=p2p)
    if f2p_passed == f2p["total"]:
        return _verdict(
            False,
            "TESTS_REGRESSION",
            f"fail_to_pass {f2p_passed}/{f2p['total']} but pass_to_pass {p2p_passed}/{p2p['total']}",
            raw,
            task,
            f2p=f2p,
            p2p=p2p,
        )
    return _verdict(
        False,
        "TESTS_FAIL",
        f"fail_to_pass {f2p_passed}/{f2p['total']} after patch; pass_to_pass {p2p_passed}/{p2p['total']}",
        raw,
        task,
        f2p=f2p,
        p2p=p2p,
    )


def _verdict(
    resolved: bool,
    error_code: str,
    detail: str,
    raw: dict,
    task: Task,
    *,
    f2p: dict | None = None,
    p2p: dict | None = None,
    ran: bool = True,
) -> Verdict:
    if f2p is None:
        f2p = {"passed": 0, "total": len(task.fail_to_pass)}
    if p2p is None:
        p2p = {"passed": 0, "total": len(task.pass_to_pass)}
    return Verdict(
        resolved=resolved,
        error_code=error_code,
        detail=_truncate(detail),
        fail_to_pass=f2p,
        pass_to_pass=p2p,
        grader=GRADER,
        grader_version=GRADER_VERSION,
        raw={"ran_tests": ran, **raw},
    )


def _pytest_version() -> str | None:
    """Version of the pytest that will actually grade, or None if it is not importable."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "unknown"


def grading_requirements() -> list[tuple[str, bool, str]]:
    """(name, present, how-to-fix) for everything grade() needs on THIS host.

    run.sh preflight tier 3 calls this before the first model call. agenttask grades by
    running the hidden tests with pytest in-process, so a missing pytest means the suite
    cannot be graded at all — and would otherwise be discovered only at the first grade(),
    reported as TESTS_FAIL 0/N, and published as if the model had failed.
    """
    reqs = []
    reqs.append((
        "pytest (the agenttask test runner)",
        _pytest_version() is not None,
        "%s -m pip install pytest" % sys.executable,
    ))
    reqs.append((
        "git (workspace base for the attempt diff)",
        _tool_version("git") is not None,
        "install git",
    ))
    return reqs


def environment_digest() -> str:
    """Identity of the grading environment (CONTRACTS.md §5). Best-effort: it degrades to a
    packless digest so a manifest can still be written on a machine without the data pack."""
    pack: dict[str, Any]
    try:
        data_dir = _data_dir()
        info = _pack_info(data_dir)
        pack = {
            "revision": info.get("pack_revision", "unresolved"),
            "sha256": info.get("pack_sha256") or _dir_digest(data_dir / "tasks"),
            "task_count": info.get("task_count"),
        }
    except (AgentTaskDataError, OSError):
        pack = {"revision": "unresolved", "sha256": None, "task_count": None}

    payload = {
        "suite": SUITE_NAME,
        "adapter_version": ADAPTER_VERSION,
        "grader": GRADER,
        "grader_version": GRADER_VERSION,
        "pack": pack,
        "python": platform.python_version(),
        "git": _tool_version("git"),
        "default_test_cmd": DEFAULT_TEST_CMD,
        # The test runner IS the grader here: without it every attempt grades TESTS_FAIL
        # with 0/N passing, which reads as a model result rather than a broken host.
        "pytest": _pytest_version(),
    }
    return "sha256:" + _sha256_bytes(_canonical_json(payload).encode("utf-8"))


# --- workspace construction -------------------------------------------------------------


def materialize(task: Task, dest: Path, *, include_hidden_tests: bool = False) -> Path:
    """Public helper for the environment layer: lay the task's repo snapshot into `dest`.

    The agent gets include_hidden_tests=False — that is what makes the tests hidden.
    grade() overlays them afterwards, on its own private copy.
    """
    _materialize_snapshot(task, dest)
    if include_hidden_tests:
        _overlay_hidden_tests(task, dest)
    return dest


def _pack_path(task: Task, spec: Mapping[str, Any], what: str) -> Path:
    data_dir = Path(task.metadata["data_dir"])
    path = (data_dir / str(spec["path"])).resolve()
    if not _is_within(path, data_dir):
        raise AgentTaskDataError(f"{task.instance_id}: {what} path escapes the data pack")
    if not path.exists():
        raise AgentTaskDataError(f"{task.instance_id}: {what} missing at {path}")
    want = spec.get("sha256")
    if want and path.is_file():
        got = _sha256_file(path)
        if got != want:
            raise AgentTaskDataError(f"{task.instance_id}: {what} sha256 mismatch (pack corrupt)")
    return path


def _materialize_snapshot(task: Task, dest: Path) -> None:
    src = _pack_path(task, task.metadata["snapshot"], "snapshot")
    dest.mkdir(parents=True, exist_ok=True)
    _unpack(src, dest, task.instance_id)


def _overlay_hidden_tests(task: Task, dest: Path) -> list[str]:
    """Unpack the grader-only tests over the workspace. Returns the paths that the agent's
    patch had also touched — overwriting them is intended (the overlay is authoritative),
    but a non-empty list is a tampering signal worth recording."""
    src = _pack_path(task, task.metadata["hidden_tests"], "hidden_tests")
    before = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    names = _unpack(src, dest, task.instance_id)
    return sorted({n for n in names if n in before})


def _unpack(src: Path, dest: Path, iid: str) -> list[str]:
    """Extract an archive (or copy a directory) into `dest`; returns the member paths."""
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        names: list[str] = []
        for path in sorted(src.rglob("*")):
            rel = path.relative_to(src)
            target = dest / rel
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                names.append(rel.as_posix())
        return names

    name = src.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(src) as zf:
            members = [m for m in zf.namelist() if _safe_member(m)]
            if len(members) != len(zf.namelist()):
                raise AgentTaskDataError(f"{iid}: archive {src.name} contains unsafe paths")
            zf.extractall(dest, members=members)
            return [m for m in members if not m.endswith("/")]
    if any(name.endswith(ext) for ext in (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        with tarfile.open(src) as tf:
            members = tf.getmembers()
            for member in members:
                if not _safe_member(member.name) or member.issym() or member.islnk():
                    raise AgentTaskDataError(f"{iid}: archive {src.name} contains unsafe entry {member.name!r}")
            try:
                tf.extractall(dest, filter="data")
            except TypeError:  # Python < 3.11.4 has no extraction filters
                tf.extractall(dest)
            return [m.name for m in members if m.isfile()]
    raise AgentTaskDataError(f"{iid}: unsupported archive format {src.name}")


def _safe_member(name: str) -> bool:
    if name.startswith("/") or name.startswith("\\"):
        return False
    parts = Path(name).parts
    return ".." not in parts and not any(p.endswith(":") for p in parts[:1])


# --- patching and test execution --------------------------------------------------------


def _apply_patch(work: Path, patch: str, tmp: Path) -> tuple[bool, str]:
    diff = tmp / "agent.diff"
    diff.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")

    attempts: list[tuple[str, list[str]]] = []
    if (work / ".git").exists():
        attempts.append(("git apply --3way", ["git", "apply", "--3way", "-p1", str(diff)]))
    attempts.append(("git apply", ["git", "apply", "-p1", str(diff)]))
    attempts.append(("patch -p1", ["patch", "-p1", "--batch", "--forward", "-i", str(diff)]))

    reasons = []
    for label, argv in attempts:
        try:
            proc = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            reasons.append(f"{label}: tool missing")
            continue
        except subprocess.TimeoutExpired:
            reasons.append(f"{label}: timeout")
            continue
        if proc.returncode == 0:
            return True, label
        reasons.append(f"{label}: rc={proc.returncode}")
    return False, "; ".join(reasons)


def _text(data: Any) -> str:
    """Bytes-or-str-or-None → str. `TimeoutExpired.stdout` is bytes on CPython < 3.10 even
    when the process was opened in text mode."""
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


def _run(cmd: str, cwd: Path, timeout_s: float) -> tuple[int, str, bool]:
    """Run `cmd` in a shell; return (returncode, combined output, timed_out).

    `timed_out` is True only when OUR deadline fired (subprocess.TimeoutExpired). A command
    that exits 124 on its own (e.g. a `timeout(1)` wrapper inside test_cmd) is reported as
    rc=124 with timed_out=False — the two must not be conflated, because the former is a
    grader hang (INFRA_GRADER) and the latter is an observed test outcome.

    The command runs in its own session so a timeout kills the whole process group, not
    just the shell — a hung pytest must not outlive its attempt on a rented GPU node.
    """
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "CI": "1"})
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=max(0.0, float(timeout_s)))
    except subprocess.TimeoutExpired as exc:
        _kill_group(proc)
        # Retrying communicate() after a timeout returns EVERYTHING captured so far (the
        # partial output on `exc` included), so it is used alone; the exception's copy is
        # only the fallback if the retry itself cannot drain the pipes.
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            stdout, stderr = exc.stdout, exc.stderr
        return 124, _text(stdout) + _text(stderr), True
    return proc.returncode, _text(stdout) + _text(stderr), False


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, 9)  # SIGKILL; the session was created by start_new_session
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _run_tests(work: Path, env: Mapping[str, Any], node_ids: list[str]) -> tuple[dict, int, str, bool]:
    """Run every node id of one attempt under ONE shared `test_timeout_s` budget.

    Returns (statuses, returncode, output, timed_out). `timed_out=True` means the per-attempt
    deadline fired (TIMEOUT_SCOPE); the caller raises GraderError for it — the statuses
    gathered so far are partial and must not be scored.
    """
    budget_s = float(env["test_timeout_s"])

    if env.get("runner") == "exit-code":
        # One process per node id, all charged against the same deadline so the scope
        # matches the pytest runner's single batched invocation.
        statuses: dict[str, str] = {}
        rc_total, chunks = 0, []
        deadline = time.monotonic() + budget_s
        for nid in node_ids:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                chunks.append(f"[agenttask] per-attempt budget of {budget_s:g}s exhausted before {nid}")
                return statuses, rc_total, "\n".join(chunks), True
            cmd = _compose_test_cmd(str(env["test_cmd"]), [nid])
            rc, out, timed_out = _run(cmd, work, remaining)
            chunks.append(out[-OUTPUT_TAIL_CHARS:])
            if timed_out:
                return statuses, rc_total, "\n".join(chunks), True
            statuses[nid] = "passed" if rc == 0 else "failed"
            rc_total |= rc
        return statuses, rc_total, "\n".join(chunks), False

    cmd = _compose_test_cmd(str(env["test_cmd"]), node_ids)
    rc, out, timed_out = _run(cmd, work, budget_s)
    if timed_out:
        return {}, rc, out, True

    report = env.get("report_json")
    if report:
        parsed = _parse_json_report(work / str(report))
        if parsed is not None:
            return parsed, rc, out, False
    return _parse_pytest_output(out), rc, out, False


def _compose_test_cmd(template: str, node_ids: list[str]) -> str:
    tests = " ".join(shlex.quote(n) for n in node_ids)
    if "{tests}" in template:
        return template.replace("{tests}", tests)
    return f"{template} {tests}"


# pytest's -rA short-summary line is `<OUTCOME> <nodeid>` optionally followed by ` - <reason>`.
# The node id is printed verbatim and MAY contain spaces (parametrize ids), so it is captured
# greedily to end of line and the reason is split off afterwards by _split_summary_rest().
_PYTEST_LINE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS)\s+(.+?)\s*$")


def _split_summary_rest(rest: str) -> str:
    """Strip a trailing ` - <reason>` from the text after the outcome, returning the node id.

    The separator is the first ` - ` that sits outside the `[...]` parametrize brackets, so
    an id such as `test_x[a - b]` survives while `test_x[a - b] - AssertionError` is cut.
    """
    depth = 0
    i = 0
    n = len(rest)
    while i < n:
        ch = rest[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == " " and depth == 0 and rest.startswith(" - ", i):
            return rest[:i]
        i += 1
    return rest


def _parse_pytest_output(out: str) -> dict[str, str]:
    """Read pytest's `-rA` short summary. Anything not reported PASSED counts as not passed.

    >>> _parse_pytest_output("PASSED tests/test_a.py::test_plain")
    {'tests/test_a.py::test_plain': 'passed'}
    >>> _parse_pytest_output("PASSED tests/test_dates.py::test_parse[2020-01-01 00:00:00-utc]")
    {'tests/test_dates.py::test_parse[2020-01-01 00:00:00-utc]': 'passed'}
    >>> _parse_pytest_output("FAILED tests/test_a.py::test_x - AssertionError: 1 - 2")
    {'tests/test_a.py::test_x': 'failed'}
    >>> _parse_pytest_output("FAILED tests/test_a.py::test_x[a - b] - assert 1 == 2")
    {'tests/test_a.py::test_x[a - b]': 'failed'}
    >>> _parse_pytest_output("ERROR tests/test_b.py::test_setup - ValueError: boom")
    {'tests/test_b.py::test_setup': 'error'}
    >>> _parse_pytest_output("ERROR tests/test_c.py - ImportError: no module")
    {'tests/test_c.py': 'error'}
    >>> _parse_pytest_output("XFAIL tests/test_a.py::test_y\\n  known bug")
    {'tests/test_a.py::test_y': 'xfail'}
    >>> _parse_pytest_output("  PASSED tests/test_a.py::test_z  \\n=== 1 passed in 0.01s ===")
    {'tests/test_a.py::test_z': 'passed'}
    >>> _parse_pytest_output("tests/test_a.py::test_q PASSED\\nPASSED\\n")
    {}
    """
    statuses: dict[str, str] = {}
    for line in out.splitlines():
        m = _PYTEST_LINE.match(line.strip())
        if not m:
            continue
        outcome = m.group(1)
        nodeid = _split_summary_rest(m.group(2)).rstrip(":").strip()
        if not nodeid:
            continue
        statuses[nodeid] = "passed" if outcome == "PASSED" else outcome.lower()
    return statuses


def _selftest_parse_pytest_output() -> None:
    """Unit test for the -rA parser (finding [8]). Run: `python3 -m doctest` on this file, or
    `python3 -c "from harness.adapters import agenttask as a; a._selftest_parse_pytest_output()"`."""
    space_id = "tests/test_dates.py::test_parse[2020-01-01 00:00:00-utc]"
    out = "\n".join([
        "=== short test summary info ===",
        "PASSED tests/test_a.py::test_plain",
        f"PASSED {space_id}",
        "FAILED tests/test_a.py::test_x - AssertionError: 1 - 2",
        "ERROR tests/test_b.py::test_setup - ValueError: boom",
        "XPASS tests/test_a.py::test_xp",
        "=== 2 passed, 1 failed, 1 error in 0.10s ===",
    ])
    got = _parse_pytest_output(out)
    assert got["tests/test_a.py::test_plain"] == "passed", got
    assert got[space_id] == "passed", got
    assert got["tests/test_a.py::test_x"] == "failed", got
    assert got["tests/test_b.py::test_setup"] == "error", got
    assert got["tests/test_a.py::test_xp"] == "xpass", got
    assert len(got) == 5, got
    assert _split_summary_rest("t[a - b] - reason") == "t[a - b]"
    assert _split_summary_rest("t[a - b]") == "t[a - b]"
    assert _split_summary_rest("t - r - s") == "t"


def _parse_json_report(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise GraderError(f"agenttask grader: unparseable test report {path.name}: {exc}") from exc
    tests = doc.get("tests")
    if not isinstance(tests, list):
        raise GraderError(f"agenttask grader: test report {path.name} has no 'tests' array")
    return {str(t.get("nodeid")): str(t.get("outcome")) for t in tests if t.get("nodeid")}


def _tool_version(tool: str) -> str:
    try:
        proc = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unresolved"
    return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "unresolved"
