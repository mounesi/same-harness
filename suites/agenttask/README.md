# AgentTask suite — 50 internal tasks

> ## ⛔ PUBLICATION IS BLOCKED ON THE CONSENT DECISION (AI-2955)
>
> These 50 tasks are derived from real customer work in AgentTask. Until AI-2955 resolves,
> **no customer-derived content may enter git history, the paper, a figure, an appendix, a
> slide, or any public artifact.** That covers task text, repo snapshots, hidden tests,
> agent patches, trajectories, and grader output.
>
> Git history is forever. A file deleted in a later commit is still published. If you are
> unsure whether something is derived from a customer repo, it is — leave it out.
>
> **Git-safe from this suite, and nothing else:**
> * `seed.json` — instance **ids only** (this directory)
> * aggregate counts and rates in `analysis/tables/` (no per-task rows that reveal task text)
>
> Everything else lives in the out-of-tree data pack (below) and, for results, in the
> restricted object-store prefix. Every run of this suite carries `consent_class:
> "restricted"` in its manifest and on every raw-result record; `resultsctl verify
> --git-hygiene` fails a commit that stages a restricted record.
>
> When AI-2955 resolves, the decision — and its scope — gets recorded here before anything
> changes.

Adapter: [`harness/adapters/agenttask.py`](../../harness/adapters/agenttask.py).
Contract: [`docs/CONTRACTS.md`](../../docs/CONTRACTS.md) §5 (adapter API), §6 (seed files),
§7.4 (what git may hold).

---

## What a task is

Same shape as SWE-bench, built from internal work instead of public GitHub issues:

| Piece | Where it lives | Who sees it |
|---|---|---|
| repo snapshot at the pre-fix commit | `snapshots/<id>.tar.gz` | the agent |
| issue text (`problem_statement`) | `tasks/<id>.json` | the agent, verbatim, unmodified |
| hidden tests | `hidden-tests/<id>.tar.gz` | **grader only** — overlaid *after* the patch |
| `fail_to_pass` / `pass_to_pass` node ids | `tasks/<id>.json` | grader only |

Grading is SWE-bench semantics: apply the agent's diff to a throwaway copy of the snapshot,
overlay the hidden tests, run setup, run the tests. `resolved` requires **every**
`fail_to_pass` test green **and** every `pass_to_pass` test still green.

The hidden-test overlay is applied after the patch, so it always wins. If the patch touched a
path the overlay also owns, the grader records that in
`raw.hidden_tests_overwrote_patched_paths` — a non-empty list is a test-tampering signal
worth a look, not an automatic failure.

## The data pack (out of tree)

The adapter refuses to read a pack that lives inside this repo's working tree. That refusal is
the consent guard, not a convenience — keep the pack somewhere git cannot reach.

Resolution order: `$AGENTTASK_DATA_DIR`, then `/persistent/agenttask`, then
`~/.harness/agenttask`.

```
<pack>/
  PACK.json                        {"schema":"agenttask-pack/v1","pack_revision":"<git sha of the
                                    internal task repo>","task_count":50,"pack_sha256":"…"}
  tasks/<instance_id>.json         one agenttask-task/v1 record (below)
  snapshots/<instance_id>.tar.gz   agent-visible repo snapshot, hidden tests removed
  hidden-tests/<instance_id>.tar.gz  grader-only overlay
```

`instance_id` must match `^[A-Za-z0-9._-]+$` — it becomes a directory name in the run tree —
and must be **opaque**: no customer name, no repo name, no ticket title. Ids are the one thing
from this suite that reaches git.

`PACK.json.pack_revision` is what makes a run reproducible: it pins the internal task repo
commit the pack was built from, and it feeds `environment_digest()`.

## Task record — `agenttask-task/v1`

Synthetic example (this file may never contain a real one):

```json
{
  "schema": "agenttask-task/v1",
  "instance_id": "at-0042",
  "repo": "",
  "origin": "internal/agenttask",
  "labels": ["backend", "bugfix"],
  "problem_statement": "Uploading a file larger than 2 GiB fails with a misleading 400 …",
  "snapshot": {
    "path": "snapshots/at-0042.tar.gz",
    "sha256": "…",
    "base_commit": "4f2e51d8b7a3c6e4d90f2a7b1c33116df67c0a9b"
  },
  "hidden_tests": {
    "path": "hidden-tests/at-0042.tar.gz",
    "sha256": "…"
  },
  "fail_to_pass": ["tests/test_upload.py::test_large_file_streams"],
  "pass_to_pass": ["tests/test_upload.py::test_small_file", "tests/test_auth.py::test_token"],
  "environment": {
    "image": "python:3.11-slim",
    "setup_cmds": ["python -m pip install -e .[test]"],
    "test_cmd": "python -m pytest -rA -p no:cacheprovider {tests}",
    "runner": "pytest",
    "report_json": null,
    "test_timeout_s": 900,
    "setup_timeout_s": 1800
  }
}
```

Field notes:

- **`problem_statement`** is handed to the model **verbatim**. No AgentTask-specific preamble,
  no hints, no reformatting — all three suites render the same prompt template
  (`harness/prompts/`), and only the variable values differ. That is the whole study: the
  harness is the control variable.
- **`repo`** is `""` for tasks with no meaningful upstream repo identity. Do not put a
  customer repo name here.
- **`fail_to_pass` must be non-empty.** A task with no red test is not gradable and
  `load_tasks()` rejects the pack.
- **`environment.test_cmd`** takes a `{tests}` placeholder; the grader substitutes the
  shell-quoted node ids. Without the placeholder they are appended.
- **`environment.runner`** is `pytest` (parse the `-rA` short summary) or `exit-code` (run
  each node id separately; exit 0 = passed). Set `report_json` to a workspace-relative
  pytest-json-report path to be parsed in preference to stdout.
- **`environment.image`** is informational — this adapter runs the commands in a temp
  directory on the host. If a task needs real container isolation, that belongs in the
  environment layer, not here.
- Anything else you add goes in `labels`/`metadata` and is ignored by the harness.

## Seed file

`seed.json` is a `suite-seed/v1` file (CONTRACTS.md §6.1) with `selection.method:
"full-enumeration"` — all 50 tasks, no sampling. The seed integer is still recorded because
the pass-level RNG uses it.

**The committed `seed.json` is a PLACEHOLDER** with synthetic ids. Regenerate it from the real
pack before any run:

```bash
python3 suites/generate_seeds.py --suite agenttask \
    --population <ids.txt from the pack> --revision <internal task repo sha>
python3 suites/generate_partitions.py --force     # re-freeze, BEFORE any run
```

`load_tasks()` refuses a placeholder seed file outright (override
`AGENTTASK_ALLOW_PLACEHOLDER=1` only for harness self-tests).

## Failure mapping

The adapter returns, never raises, for task-level outcomes (CONTRACTS.md §4):

| Situation | `error_code` |
|---|---|
| empty/whitespace diff | `NO_PATCH` |
| `git apply --3way` → `git apply` → `patch -p1` all fail | `PATCH_MALFORMED` |
| snapshot/overlay missing, sha mismatch, unsafe archive, setup command failed | `INFRA_SANDBOX` |
| some `fail_to_pass` still red, **or the test run hit `test_timeout_s`** | `TESTS_FAIL` |
| all `fail_to_pass` green but a `pass_to_pass` broke | `TESTS_REGRESSION` |
| all green | `OK` (`resolved: true`) |
| grader machinery itself broken (unparseable report, temp-dir IO) | raises `GraderError` → `INFRA_GRADER` |

A test-run timeout is charged to the model (`TESTS_FAIL`), not to infrastructure: the patch is
the only thing that varies, and an infinite loop introduced by the patch is a real failure.
Setup timeouts are charged to infrastructure (`INFRA_SANDBOX`) and dropped from the
denominator.
