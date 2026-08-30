"""SWE-bench Verified adapter — CONTRACTS.md §5.

Suite: `swebench-verified`, a 100-instance seeded subset of the 500-instance
`princeton-nlp/SWE-bench_Verified` test split.  The subset lives in
`suites/verified-100.json` and is authoritative: this module never samples.

Task rows are pulled at run time from HuggingFace, pinned to the dataset
revision recorded in the seed file.  Nothing from the benchmark is vendored into
this repo.

Grading runs the official SWE-bench evaluation harness in the instance's own
container; see the "HOW EVALUATION IS INVOKED" block in
`harness/adapters/_swebench.py`.

All real work lives in `_swebench.py`, which `swebench_pro.py` also uses: the
two suites share one grading implementation and differ only in the `SuiteSpec`
below.
"""

from __future__ import annotations

import sys
from pathlib import Path

from harness.adapters import _swebench
from harness.types import Prompt, Task, Verdict

SUITE_NAME = "swebench-verified"
ADAPTER_VERSION = "1.0.0"
CONSENT_CLASS = "public"

SPEC = _swebench.SuiteSpec(
    suite_name=SUITE_NAME,
    adapter_version=ADAPTER_VERSION,
    consent_class=CONSENT_CLASS,
    dataset="princeton-nlp/SWE-bench_Verified",
    split="test",
    default_seed_file="suites/verified-100.json",
    grader="swebench-eval",
    eval_module="swebench.harness.run_evaluation",
    grader_distribution="swebench",
    env_infix="SWEBENCH_VERIFIED",
    # Official image naming: double underscores are illegal in image tags, so
    # upstream substitutes `_1776_`.  Informational only — the evaluation
    # harness resolves the image itself.
    image_template="swebench/sweb.eval.x86_64.{norm_id}:latest",
    image_row_keys=(),
)


def load_tasks(seed_file: Path, partitions_file: Path | None = None) -> list[Task]:
    """Load the 100 seeded instances, in seed-file order."""
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
