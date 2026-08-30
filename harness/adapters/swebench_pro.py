"""SWE-bench Pro adapter — CONTRACTS.md §5.

Suite: `swebench-pro`, a ~50-instance seeded subset of the public SWE-bench Pro
test split.  The subset lives in `suites/pro-50.json` and is authoritative.

This module is intentionally a near-duplicate of `swebench_verified.py`: the two
suites share the same record shape (`instance_id`, `repo`, `base_commit`,
`problem_statement`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `test_patch`) and the same
resolution criterion, so they share one implementation in `_swebench.py` and one
prompt template.  Holding grading constant across the two suites is part of the
study design — a divergence here would confound the model comparison.

The three places they legitimately differ:

  * `dataset` — a different HuggingFace dataset id and revision pin.
  * `grader` — recorded as `swebench-pro-eval` so verdicts are attributable to
    the Pro evaluation path in the raw records.
  * container images — Pro instances carry their own image references in the
    dataset row, so `image_row_keys` is consulted and there is no name template.

If upstream Pro tooling ever needs a different entry point than
`swebench.harness.run_evaluation`, point `HARNESS_EVAL_CMD_SWEBENCH_PRO` at it
(a shlex-split command template supporting `{dataset} {split} {predictions}
{instance_id} {run_id} {report_dir} {timeout}`) rather than forking the grading
code — the override is recorded in `environment_digest()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from harness.adapters import _swebench
from harness.types import Prompt, Task, Verdict

SUITE_NAME = "swebench-pro"
ADAPTER_VERSION = "1.0.0"
CONSENT_CLASS = "public"

SPEC = _swebench.SuiteSpec(
    suite_name=SUITE_NAME,
    adapter_version=ADAPTER_VERSION,
    consent_class=CONSENT_CLASS,
    dataset="ScaleAI/SWE-bench_Pro",
    split="test",
    default_seed_file="suites/pro-50.json",
    grader="swebench-pro-eval",
    eval_module="swebench.harness.run_evaluation",
    grader_distribution="swebench",
    env_infix="SWEBENCH_PRO",
    # Pro instances reference their own prebuilt images; there is no stable
    # name template, so the row wins and an empty string means "the evaluation
    # harness resolves it".
    image_template=None,
    image_row_keys=("docker_image", "image_name", "instance_image_tag", "image"),
)


def load_tasks(seed_file: Path, partitions_file: Path | None = None) -> list[Task]:
    """Load the 50 seeded instances, in seed-file order."""
    return _swebench.load_tasks(SPEC, seed_file, partitions_file)


def build_prompt(task: Task) -> Prompt:
    """Render the one shared prompt template (identical id across all suites)."""
    return _swebench.build_prompt(SPEC, task)


def grade(task: Task, patch: str) -> Verdict:
    """Run the instance's FAIL_TO_PASS / PASS_TO_PASS tests against `patch`."""
    return _swebench.grade(SPEC, task, patch)


def environment_digest() -> str:
    """`sha256:…` over the grading environment (dataset, grader artifact, runtime)."""
    return _swebench.environment_digest(SPEC)


if __name__ == "__main__":
    sys.exit(_swebench.main(SPEC))
