"""Shared value types for the harness — CONTRACTS.md §5.1–§5.3.

Stdlib only, frozen dataclasses. Every adapter and the agent loop import these; nothing
here knows about a specific suite or model, which is what keeps the harness constant.

NOTE: this module shadows the stdlib ``types`` module for anything under ``harness/``
started by path. Entry points must use ``python3 -m harness.<mod>`` so absolute imports
resolve correctly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from typing import Any, Dict, Tuple

__all__ = [
    "Task",
    "Prompt",
    "Verdict",
    "GraderError",
    "PARTITIONS",
    "canonical_json",
    "canonical_sha256",
]

# Python 3.10 introduced dataclass slots=True. The CI instances run 3.11+, but the
# repo is also read on older interpreters (a 3.7 was observed in verification), so
# degrade to a plain frozen dataclass rather than raising TypeError at import time.
_SLOTS: Dict[str, bool] = {"slots": True} if sys.version_info >= (3, 10) else {}

PARTITIONS = ("train", "dev", "final_holdout", "unpartitioned")


def canonical_json(obj: Any) -> str:
    """Canonical JSON used for every hash in the study: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


class GraderError(RuntimeError):
    """Grading *infrastructure* failed — not a task-level failure.

    Adapters raise this only when the verdict is unknowable (docker unavailable,
    eval harness crashed, report unparseable). The caller maps it to INFRA_GRADER,
    which is excluded from the resolution-rate denominator (CONTRACTS §4).
    A task that simply did not resolve MUST return a Verdict instead.
    """


@dataclasses.dataclass(frozen=True, **_SLOTS)
class Task:
    """One benchmark instance, normalized across suites (CONTRACTS §5.1)."""

    suite: str
    instance_id: str
    qualified_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: Tuple[str, ...]
    pass_to_pass: Tuple[str, ...]
    environment: Dict[str, Any]
    partition: str
    metadata: Dict[str, Any]
    source_sha256: str

    def __post_init__(self) -> None:
        expected = "{}::{}".format(self.suite, self.instance_id)
        if self.qualified_id != expected:
            raise ValueError(
                "qualified_id {!r} does not match {!r}".format(self.qualified_id, expected)
            )
        if self.partition not in PARTITIONS:
            raise ValueError(
                "partition {!r} not in {}".format(self.partition, PARTITIONS)
            )


@dataclasses.dataclass(frozen=True, **_SLOTS)
class Prompt:
    """A rendered prompt (CONTRACTS §5.2).

    Harness-constant invariant: every suite renders the SAME template_id from the same
    template files; only ``variables`` differ. run.sh asserts template_id against the
    manifest on the first task of a run.
    """

    template_id: str
    system: str
    user: str
    tools: Tuple[Dict[str, Any], ...]
    prompt_sha256: str
    variables: Dict[str, Any]

    @staticmethod
    def compute_sha256(
        template_id: str, system: str, user: str, tools: Tuple[Dict[str, Any], ...]
    ) -> str:
        return canonical_sha256(
            {
                "template_id": template_id,
                "system": system,
                "user": user,
                "tools": list(tools),
            }
        )


@dataclasses.dataclass(frozen=True, **_SLOTS)
class Verdict:
    """The grader's decision on one attempt (CONTRACTS §5.3)."""

    resolved: bool
    error_code: str
    detail: str
    fail_to_pass: Dict[str, int]
    pass_to_pass: Dict[str, int]
    grader: str
    grader_version: str
    raw: Dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.detail) > 512:
            object.__setattr__(self, "detail", self.detail[:509] + "...")
