"""Adapter registry — CONTRACTS.md §5.4.

`run.sh` resolves the module path for the manifest's `harness.adapter` field
through this mapping, so it always names the file that actually graded the run.

Adapters are imported eagerly but tolerantly: an adapter whose module or
dependencies are missing is recorded in `IMPORT_ERRORS` instead of taking down
the whole registry, and `get()` re-raises with the original cause.  No adapter
imports another adapter; `_swebench` is a shared base module, not an adapter,
and is deliberately absent from `ADAPTERS`.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType

__all__ = [
    "SUITES",
    "ADAPTERS",
    "IMPORT_ERRORS",
    "MODULE_NAMES",
    "DEFAULT_SEED_FILES",
    "CONSENT_CLASSES",
    "get",
    "module_path",
    "default_seed_file",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Suite names in the canonical `--suite all` execution order (CONTRACTS §1.2).
SUITES: tuple[str, ...] = ("swebench-verified", "swebench-pro", "agenttask")

MODULE_NAMES: dict[str, str] = {
    "swebench-verified": "swebench_verified",
    "swebench-pro": "swebench_pro",
    "agenttask": "agenttask",
}

DEFAULT_SEED_FILES: dict[str, str] = {
    "swebench-verified": "suites/verified-100.json",
    "swebench-pro": "suites/pro-50.json",
    "agenttask": "suites/agenttask/seed.json",
}

CONSENT_CLASSES: dict[str, str] = {
    "swebench-verified": "public",
    "swebench-pro": "public",
    "agenttask": "restricted",
}

ADAPTERS: dict[str, ModuleType] = {}
IMPORT_ERRORS: dict[str, BaseException] = {}

for _suite, _module_name in MODULE_NAMES.items():
    try:
        _module = import_module(f"{__name__}.{_module_name}")
    except Exception as _exc:  # missing module, missing dependency, syntax error
        IMPORT_ERRORS[_suite] = _exc
        continue
    declared = getattr(_module, "SUITE_NAME", None)
    if declared != _suite:
        IMPORT_ERRORS[_suite] = RuntimeError(
            f"{_module_name}.SUITE_NAME is {declared!r}, expected {_suite!r}"
        )
        continue
    ADAPTERS[_suite] = _module

del _suite, _module_name


def get(suite: str) -> ModuleType:
    """Return the adapter module for `suite`, or raise a clear error."""
    module = ADAPTERS.get(suite)
    if module is not None:
        return module
    cause = IMPORT_ERRORS.get(suite)
    if cause is not None:
        raise ImportError(
            f"adapter for suite {suite!r} ({module_path(suite)}) could not be imported: {cause}"
        ) from cause
    raise KeyError(f"unknown suite {suite!r}; known suites: {', '.join(SUITES)}")


def module_path(suite: str) -> str:
    """Repo-relative POSIX path of the adapter module, for the run manifest."""
    if suite not in MODULE_NAMES:
        raise KeyError(f"unknown suite {suite!r}; known suites: {', '.join(SUITES)}")
    module = ADAPTERS.get(suite)
    if module is not None and getattr(module, "__file__", None):
        try:
            return Path(module.__file__).resolve().relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            return Path(module.__file__).as_posix()
    return f"harness/adapters/{MODULE_NAMES[suite]}.py"


def default_seed_file(suite: str) -> str:
    """Repo-relative default seed file for `suite` (overridable with --seed-file)."""
    try:
        return DEFAULT_SEED_FILES[suite]
    except KeyError:
        raise KeyError(f"unknown suite {suite!r}; known suites: {', '.join(SUITES)}") from None
