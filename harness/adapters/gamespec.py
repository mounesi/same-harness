#!/usr/bin/env python3
"""gamespec.py — suite adapter for the build-a-game specs (CONTRACTS.md §5).

Greenfield counterpart to the three brownfield suites: the task hands the model a written
game specification and an EMPTY workspace, and grades whatever it builds with
`suites/gamespec/floor_check.py`. There is no upstream dataset, no docker image and no
network: every task is a spec file under `suites/gamespec/specs/<instance_id>.md`, and the
grader is one Python file plus `node`.

Why the workspace is not quite empty. `capture_patch()` diffs the work tree against a base
commit, so the model's `game.html` reaches grade() as a new-file diff. The base tree holds
three files the model may read but has no reason to change:

    SPEC.md          the same text as the prompt's problem statement, for `read_file`
    floor_check.py   a copy of the floor check, so `run_tests` really runs it
    README.md        one paragraph: build game.html here, run the floor check
    .gitignore       __pycache__/ and *.pyc — running the floor check writes a pycache into
                     the workspace, and without this it lands in the captured diff as a
                     binary hunk that `git apply` then refuses (first live run, AI-3230)

grade() never uses the workspace copy of floor_check.py — a model could edit it. It always
runs the repo's own `suites/gamespec/floor_check.py`, and environment_digest() hashes that
file, so a change to the grader changes the digest as §2.3 requires.

Verdict mapping (§5.3): every floor check that fails is a `fail_to_pass` miss; `resolved`
means the floor passed in full. Passing the floor says nothing about whether the game is
any good — that is the human channel in suites/gamespec/README.md, downstream of the
sealed bundle, and it never touches `Verdict`.

CONSENT_CLASS is "public": the specs are written here, so nothing in this suite is
customer-derived.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.types import (  # noqa: E402
    PROMPT_VARIABLE_SOURCES_KEY,
    GraderError,
    Prompt,
    Task,
    Verdict,
)

SUITE_NAME = "gamespec"
ADAPTER_VERSION = "1.1.0"
CONSENT_CLASS = "public"

GRADER = "gamespec-floor"
GRADER_VERSION = "1.0.0"

SEED_SCHEMA = "suite-seed/v1"
FLOOR_SCHEMA = "gamespec-floor/v1"

SUITE_DIR = _REPO_ROOT / "suites" / "gamespec"
SPECS_DIR = SUITE_DIR / "specs"
FLOOR_CHECK = SUITE_DIR / "floor_check.py"
DEFAULT_SEED_FILE = SUITE_DIR / "seed.json"
DEFAULT_PARTITIONS_FILE = SUITE_DIR / "partitions.json"

INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# The deliverable the spec names. grade() looks for exactly this path after the patch.
DELIVERABLE = "game.html"

# The interpreter the test command and the grader both invoke. Same rule as agenttask
# (AI-3155): preflight probes the interpreter the SHELL resolves, not sys.executable.
GRADER_PYTHON = "python3"
DEFAULT_TEST_CMD = GRADER_PYTHON + " floor_check.py " + DELIVERABLE
FLOOR_TIMEOUT_S = 180
DETAIL_MAX = 512

# §5.1 prompt_variable_sources — the closed vocabulary for where each prompt variable came
# from. Compared across runs, so a spelling here is a raw-result change.
SOURCE_SPEC_FILE = "spec_file"              # suites/gamespec/specs/<id>.md supplied it
SOURCE_ADAPTER_DEFAULT = "adapter_default"  # DEFAULT_TEST_CMD, this module's constant
SOURCE_ABSENT = "absent"                    # nothing supplies it; renders as EMPTY_VALUE
PROBLEM_STATEMENT_SOURCES = (SOURCE_SPEC_FILE,)
TEST_CMD_SOURCES = (SOURCE_ADAPTER_DEFAULT,)
REPO_SOURCES = (SOURCE_ABSENT,)

WORKSPACE_README = """# {iid}

Build the deliverable described in `SPEC.md` as a single file, `{deliverable}`, in this
directory. Nothing else exists here yet — this is a greenfield task.

The automated floor is `floor_check.py` (also quoted in SPEC.md §5); the test command is:

    {test_cmd}

It exits 0 when every check passes. Only `{deliverable}` is graded.
"""


# Keeps the model's own test runs out of its patch. Only the deliverable is graded, so
# nothing legitimate is lost; what IS lost is a binary hunk that broke `git apply` on the
# first live run.
WORKSPACE_GITIGNORE = "__pycache__/\n*.pyc\n"


class GameSpecDataError(RuntimeError):
    """A spec, seed or partitions file is missing or malformed."""


# --- small helpers ----------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _truncate(text: str, limit: int = DETAIL_MAX) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _template_id() -> str:
    from harness import prompts as harness_prompts

    return harness_prompts.TEMPLATE_ID


def _probe(argv_str: str) -> str | None:
    """First output line of `argv_str` run THROUGH A SHELL, or None if it cannot run.

    Through a shell on purpose: the test command and the grader both resolve their program
    in a shell, so a probe that bypasses it measures a different thing (AI-3155)."""
    try:
        out = subprocess.run(argv_str, shell=True, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else "unknown"


# --- seed / partitions ------------------------------------------------------------------


def _load_seed(seed_file: Path) -> list[str]:
    if not seed_file.is_file():
        raise GameSpecDataError(f"seed file not found: {seed_file}")
    doc = json.loads(seed_file.read_text(encoding="utf-8"))
    if doc.get("schema") != SEED_SCHEMA:
        raise GameSpecDataError(f"{seed_file}: schema is not {SEED_SCHEMA}")
    if doc.get("suite") != SUITE_NAME:
        raise GameSpecDataError(
            f"{seed_file}: suite is {doc.get('suite')!r}, expected {SUITE_NAME!r}"
        )
    ids = list(doc.get("instance_ids") or [])
    if not ids:
        raise GameSpecDataError(f"{seed_file}: instance_ids is empty")
    if len(set(ids)) != len(ids):
        raise GameSpecDataError(f"{seed_file}: instance_ids contains duplicates")
    if doc.get("count") != len(ids):
        raise GameSpecDataError(
            f"{seed_file}: count {doc.get('count')} != len(instance_ids) {len(ids)}"
        )
    for iid in ids:
        if not INSTANCE_ID_RE.match(iid):
            raise GameSpecDataError(f"{seed_file}: instance_id {iid!r} is not ^[A-Za-z0-9._-]+$")
    want = doc.get("instance_ids_sha256")
    got = _sha256_bytes(("\n".join(sorted(ids)) + "\n").encode("utf-8"))
    if want and want != got:
        raise GameSpecDataError(f"{seed_file}: instance_ids_sha256 mismatch ({want} != {got})")
    return ids


def _default_partitions_path() -> Path | None:
    env = os.environ.get("HARNESS_PARTITIONS", "").strip()
    if env:
        return Path(env)
    return DEFAULT_PARTITIONS_FILE if DEFAULT_PARTITIONS_FILE.is_file() else None


def _partition_map(partitions: Any) -> dict[str, str]:
    """qualified_id -> partition name. An absent file means every task is "unpartitioned"."""
    if partitions is None:
        return {}
    if isinstance(partitions, Mapping):
        return dict(partitions)
    path = Path(partitions)
    if not path.is_file():
        raise GameSpecDataError(f"partitions file not found: {path}")
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
    """One Task per seed id, in seed order. The spec file IS the task record."""
    seed_file = Path(seed_file)
    ids = _load_seed(seed_file)
    part = _partition_map(
        partitions_file if partitions_file is not None else _default_partitions_path()
    )
    missing = [i for i in ids if not (SPECS_DIR / f"{i}.md").is_file()]
    if missing:
        raise GameSpecDataError(
            f"{len(missing)} of {len(ids)} spec files missing from {SPECS_DIR} "
            f"(first: {missing[0]}.md)"
        )
    return [_build_task(iid, part) for iid in ids]


def _build_task(iid: str, part: Mapping[str, str]) -> Task:
    spec_path = SPECS_DIR / f"{iid}.md"
    spec_text = spec_path.read_text(encoding="utf-8")
    if not spec_text.strip():
        raise GameSpecDataError(f"{spec_path} is empty")
    qid = f"{SUITE_NAME}::{iid}"
    record = {
        "schema": "gamespec-task/v1",
        "instance_id": iid,
        "spec_sha256": _sha256_file(spec_path),
        "deliverable": DELIVERABLE,
    }
    return Task(
        suite=SUITE_NAME,
        instance_id=iid,
        qualified_id=qid,
        repo="",
        base_commit="",
        problem_statement=spec_text,
        fail_to_pass=("floor",),
        pass_to_pass=(),
        environment={
            "image": "",
            "setup_cmds": [],
            "test_cmd": DEFAULT_TEST_CMD,
            "deliverable": DELIVERABLE,
        },
        partition=part.get(qid, "unpartitioned"),
        metadata={
            "origin": "suites/gamespec/specs",
            "spec_path": str(spec_path.relative_to(_REPO_ROOT)),
            "spec_sha256": record["spec_sha256"],
            PROMPT_VARIABLE_SOURCES_KEY: {
                "problem_statement": SOURCE_SPEC_FILE,
                "repo": SOURCE_ABSENT,
                "test_cmd": SOURCE_ADAPTER_DEFAULT,
            },
        },
        source_sha256=_sha256_bytes(_canonical_json(record).encode("utf-8")),
    )


def build_prompt(task: Task) -> Prompt:
    """Render the ONE shared template — same id, same files, only the values differ."""
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
        raise GameSpecDataError(
            f"prompt template drift: rendered {prompt.template_id!r}, expected {template_id!r}"
        )
    return prompt


def materialize(task: Task, dest: Path, *, include_hidden_tests: bool = False) -> Path:
    """Lay down the base tree the model starts from (agent.py build_workspace() hook).

    `include_hidden_tests` is accepted for signature parity with agenttask and ignored:
    this suite has no hidden tests — the floor check is public by design (spec §5)."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise GameSpecDataError(f"materialize target is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SPEC.md").write_text(task.problem_statement, encoding="utf-8")
    shutil.copyfile(FLOOR_CHECK, dest / "floor_check.py")
    (dest / ".gitignore").write_text(WORKSPACE_GITIGNORE, encoding="utf-8")
    (dest / "README.md").write_text(
        WORKSPACE_README.format(
            iid=task.instance_id,
            deliverable=task.environment.get("deliverable", DELIVERABLE),
            test_cmd=task.environment.get("test_cmd", DEFAULT_TEST_CMD),
        ),
        encoding="utf-8",
    )
    return dest


def grade(task: Task, patch: str) -> Verdict:
    """Apply `patch` to a fresh base tree and run the repo's floor check on the deliverable.

    Returns (never raises) for task-level failures. Raises GraderError only when the grading
    machinery itself is broken (no node, floor_check.py crashed, unparseable output)."""
    if not patch or not patch.strip():
        return _verdict(False, "NO_PATCH", "agent produced no diff", {}, ran=False)
    base = os.environ.get("GAMESPEC_WORKDIR") or None
    try:
        with tempfile.TemporaryDirectory(prefix="gamespec-grade-", dir=base) as tmp:
            return _grade_in(Path(tmp), task, patch)
    except GraderError:
        raise
    except OSError as exc:
        raise GraderError(f"gamespec grader io error: {exc}") from exc


def _grade_in(tmp: Path, task: Task, patch: str) -> Verdict:
    work = tmp / "work"
    materialize(task, work)
    _git_base(work)

    applied, how = _apply_patch(work, patch, tmp)
    if not applied:
        return _verdict(False, "PATCH_MALFORMED", f"patch did not apply ({how})", {}, ran=False)

    deliverable = work / task.environment.get("deliverable", DELIVERABLE)
    if not deliverable.is_file():
        return _verdict(
            False, "TESTS_FAIL",
            f"{deliverable.name} was not produced — the patch touches "
            f"{_touched(patch) or 'nothing'}",
            {"deliverable_present": False}, f2p={"passed": 0, "total": 1},
        )

    report = _run_floor(deliverable)
    checks = report.get("checks") or []
    failed = [c for c in checks if not c.get("ok")]
    passed = bool(report.get("passed")) and not failed
    detail = (
        "floor passed (%d/%d checks)" % (len(checks), len(checks))
        if passed
        else "floor failed %d/%d: %s" % (
            len(failed), len(checks),
            "; ".join(str(c.get("name")) for c in failed[:4]),
        )
    )
    return _verdict(
        passed, "OK" if passed else "TESTS_FAIL", detail,
        {"deliverable_present": True, "deliverable_bytes": deliverable.stat().st_size,
         "floor": report},
        f2p={"passed": 1 if passed else 0, "total": 1},
    )


def _run_floor(deliverable: Path) -> dict:
    """The repo's floor_check.py, never the workspace copy. Exit 2 = grader broken."""
    cmd = "%s %s %s --json" % (GRADER_PYTHON, _shq(str(FLOOR_CHECK)), _shq(str(deliverable)))
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=str(deliverable.parent),
            capture_output=True, text=True, timeout=FLOOR_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise GraderError(f"floor_check.py exceeded {FLOOR_TIMEOUT_S}s") from exc
    except OSError as exc:
        raise GraderError(f"could not run floor_check.py: {exc}") from exc
    if proc.returncode == 2:
        raise GraderError("floor_check.py refused to run: %s" % (proc.stderr or "").strip()[-300:])
    try:
        report = json.loads(proc.stdout or "")
    except ValueError as exc:
        raise GraderError(
            "floor_check.py produced no JSON (rc=%d): %s"
            % (proc.returncode, ((proc.stderr or proc.stdout) or "").strip()[-300:])
        ) from exc
    if report.get("schema") != FLOOR_SCHEMA:
        raise GraderError(f"floor_check.py report schema is {report.get('schema')!r}, expected {FLOOR_SCHEMA!r}")
    return report


def _verdict(
    resolved: bool, error_code: str, detail: str, raw: dict, *,
    f2p: dict | None = None, ran: bool = True,
) -> Verdict:
    return Verdict(
        resolved=resolved,
        error_code=error_code,
        detail=_truncate(detail),
        fail_to_pass=f2p if f2p is not None else {"passed": 0, "total": 1},
        pass_to_pass={"passed": 0, "total": 0},
        grader=GRADER,
        grader_version=GRADER_VERSION,
        raw={"ran_tests": ran, **raw},
    )


def _shq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _touched(patch: str) -> str:
    files = re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)
    return ", ".join(files[:5])


def _git_base(work: Path) -> None:
    """Commit the base tree so `git apply --3way` has an index to work against."""
    cmds = (
        ["git", "init", "-q", "."],
        ["git", "config", "user.email", "harness@local"],
        ["git", "config", "user.name", "harness"],
        ["git", "add", "-A", "--", "."],
        ["git", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "harness-base", "--allow-empty"],
    )
    for argv in cmds:
        try:
            proc = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GraderError(f"git unavailable for grading: {exc}") from exc
        if proc.returncode != 0:
            raise GraderError(f"{' '.join(argv)} failed: {(proc.stderr or proc.stdout).strip()[-200:]}")


def _apply_patch(work: Path, patch: str, tmp: Path) -> tuple[bool, str]:
    diff = tmp / "agent.diff"
    diff.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")
    attempts = [
        ("git apply --3way", ["git", "apply", "--3way", "-p1", str(diff)]),
        ("git apply", ["git", "apply", "-p1", str(diff)]),
        ("patch -p1", ["patch", "-p1", "--batch", "--forward", "-i", str(diff)]),
    ]
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


# --- preflight / provenance -------------------------------------------------------------


def _node_version() -> str | None:
    return _probe("node --version")


def _grader_python_version() -> str | None:
    return _probe("%s --version" % GRADER_PYTHON)


def grading_requirements() -> list[tuple[str, bool, str]]:
    """(name, present, how-to-fix) for everything grade() needs on THIS host."""
    return [
        ("%s (the interpreter floor_check.py runs under)" % GRADER_PYTHON,
         _grader_python_version() is not None,
         "install python3, or put it on PATH for the shell that runs the harness"),
        ("node (floor_check.py drives SimCore headlessly in Node)",
         _node_version() is not None,
         "install Node.js (any current LTS) and make sure `node --version` works in a shell"),
        ("git (workspace base for the attempt diff, and patch application)",
         _probe("git --version") is not None,
         "install git"),
        ("suites/gamespec/floor_check.py", FLOOR_CHECK.is_file(),
         "the floor check is missing from the checkout"),
    ]


def environment_digest() -> str:
    """Identity of the grading environment (§2.3): the grader file, the specs, node, python."""
    specs = {}
    if SPECS_DIR.is_dir():
        for p in sorted(SPECS_DIR.glob("*.md")):
            specs[p.name] = _sha256_file(p)
    payload = {
        "suite": SUITE_NAME,
        "adapter_version": ADAPTER_VERSION,
        "grader": GRADER,
        "grader_version": GRADER_VERSION,
        "floor_check_sha256": _sha256_file(FLOOR_CHECK) if FLOOR_CHECK.is_file() else None,
        "specs": specs,
        "python": platform.python_version(),
        "grader_python": _grader_python_version(),
        "node": _node_version(),
        "default_test_cmd": DEFAULT_TEST_CMD,
    }
    return "sha256:" + _sha256_bytes(_canonical_json(payload).encode("utf-8"))


# --- CLI (development aid) --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="gamespec adapter — list tasks or grade a diff")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("tasks")
    ls.add_argument("--seed-file", default=str(DEFAULT_SEED_FILE))
    gr = sub.add_parser("grade")
    gr.add_argument("instance_id")
    gr.add_argument("patch", help="path to a unified diff, or - for stdin")
    dg = sub.add_parser("digest")
    rb = sub.add_parser("rebuild", help="lay down the base tree, apply a diff, print the deliverable path")
    rb.add_argument("instance_id")
    rb.add_argument("patch", help="path to a unified diff")
    rb.add_argument("out_dir", help="directory to build into (must not exist or be empty)")
    args = ap.parse_args(argv)
    if args.cmd == "tasks":
        for t in load_tasks(Path(args.seed_file)):
            print(t.qualified_id, t.partition, t.metadata["spec_sha256"][:12])
        return 0
    if args.cmd == "digest":
        print(environment_digest())
        return 0
    tasks = {t.instance_id: t for t in load_tasks()}
    task = tasks.get(args.instance_id)
    if task is None:
        print("unknown instance %r; have %s" % (args.instance_id, sorted(tasks)), file=sys.stderr)
        return 2
    patch = sys.stdin.read() if args.patch == "-" else Path(args.patch).read_text(encoding="utf-8")
    if args.cmd == "rebuild":
        out = Path(args.out_dir)
        materialize(task, out)
        _git_base(out)
        with tempfile.TemporaryDirectory(prefix="gamespec-rebuild-") as tmp:
            applied, how = _apply_patch(out, patch, Path(tmp))
        if not applied:
            print("patch did not apply (%s)" % how, file=sys.stderr)
            return 1
        deliverable = out / task.environment.get("deliverable", DELIVERABLE)
        if not deliverable.is_file():
            print("patch applied (%s) but produced no %s" % (how, deliverable.name), file=sys.stderr)
            return 1
        print(deliverable)
        return 0
    v = grade(task, patch)
    print(json.dumps({"resolved": v.resolved, "error_code": v.error_code, "detail": v.detail,
                      "fail_to_pass": v.fail_to_pass}, indent=2))
    return 0 if v.resolved else 1


if __name__ == "__main__":
    sys.exit(main())
