"""harness.prompts — the ONE prompt, versioned and hashed.

This package is the study's control variable made concrete: every suite adapter renders
the *same* template id from the *same* files here (CONTRACTS.md §5.2), so the only thing
that differs between two runs of two models is the model.

Files in this directory (all of them covered by ``prompt_dir_sha256`` in the run manifest,
computed by :func:`directory_digest` per CONTRACTS.md §2.4):

    TEMPLATE_ID    the template id, one line, e.g. ``agent-v1``
    system.md      the system prompt — no placeholders, identical for every task
    user.md        the user message template — ``{{placeholder}}`` substitution
    tools.json     the OpenAI tool schemas handed to every model, identical across suites
    notices.json   the fixed interstitial messages the harness itself injects
    __init__.py    this file

Changing any of these changes a verdict, so it is a MAJOR harness version bump and the new
runs are not comparable with the old ones (CONTRACTS.md, Versioning table).

Public API::

    render(template_id, variables) -> Prompt
    TEMPLATE_ID, PROMPTS_DIR, system_text(), user_template(), tools(), notices()
    directory_digest(root) -> hex        # CONTRACTS.md §2.4
    dir_sha256() -> hex                  # directory_digest(PROMPTS_DIR)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "PromptError",
    "Prompt",
    "TEMPLATE_ID",
    "PROMPTS_DIR",
    "render",
    "system_text",
    "user_template",
    "tools",
    "notices",
    "notice",
    "directory_digest",
    "dir_sha256",
    "canonical_json",
]

PROMPTS_DIR = Path(__file__).resolve().parent

#: Files that make up the template. Kept explicit so a stray file in this directory is a
#: visible problem rather than a silent change of meaning.
TEMPLATE_FILES = ("TEMPLATE_ID", "system.md", "user.md", "tools.json", "notices.json")

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")

#: Rendered in place of a variable whose value is empty or whitespace-only, so that a suite
#: with no repo (agenttask synthetic tasks) still produces the same prompt *structure* as
#: SWE-bench. Structure is part of the control variable.
EMPTY_VALUE = "(not specified)"


class PromptError(Exception):
    """Raised for a template/variable mismatch. Callers map this to a config error."""


try:  # the shared dataclasses live in harness/types.py; this package must not depend on it
    from harness.types import Prompt  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - exercised only when types.py is absent

    @dataclasses.dataclass(frozen=True)
    class Prompt:  # type: ignore[no-redef]
        """Fallback mirror of CONTRACTS.md §5.2 ``Prompt``.

        Field names, order and types match ``harness.types.Prompt`` exactly; this exists so
        that ``harness.prompts`` is usable (and testable) on its own.
        """

        template_id: str
        system: str
        user: str
        tools: tuple
        prompt_sha256: str
        variables: dict


def canonical_json(obj: Any) -> bytes:
    """Canonical in-memory JSON encoding used for every ``*_sha256`` over an object.

    Compact separators + sorted keys + UTF-8, matching the JSONL form in CONTRACTS.md §0.
    (The *on-disk* form for committed JSON is ``indent=2, sort_keys=True``; that one is for
    human diffing and is never hashed as an object.)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _read_text(name: str) -> str:
    path = PROMPTS_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"prompt file missing or unreadable: {path} ({exc})") from exc


def _read_json(name: str) -> Any:
    raw = _read_text(name)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptError(f"prompt file is not valid JSON: {PROMPTS_DIR / name} ({exc})") from exc


TEMPLATE_ID = _read_text("TEMPLATE_ID").strip()
if not TEMPLATE_ID:
    raise PromptError(f"{PROMPTS_DIR / 'TEMPLATE_ID'} is empty")


def system_text() -> str:
    """The system prompt. Constant for every task, every suite and every model."""
    return _read_text("system.md")


def user_template() -> str:
    """The user-message template, with ``{{placeholder}}`` variables still in place."""
    return _read_text("user.md")


def tools() -> tuple[dict, ...]:
    """The OpenAI tool schemas offered to every model, in a fixed order."""
    data = _read_json("tools.json")
    if not isinstance(data, list) or not data:
        raise PromptError("tools.json must be a non-empty JSON array")
    return tuple(data)


def notices() -> dict[str, str]:
    """Fixed messages the harness injects itself (nudges, compaction marker, ...)."""
    data = _read_json("notices.json")
    if not isinstance(data, dict):
        raise PromptError("notices.json must be a JSON object")
    return data


def notice(key: str) -> str:
    """One notice by key. Unknown keys are a programming error, not a runtime condition."""
    try:
        return notices()[key]
    except KeyError as exc:
        raise PromptError(f"unknown notice '{key}' (have: {sorted(notices())})") from exc


def placeholders(text: str) -> set[str]:
    """The ``{{name}}`` variables referenced by a template."""
    return set(_PLACEHOLDER_RE.findall(text))


def _substitute(template: str, values: dict[str, str]) -> str:
    """Single-pass ``{{name}}`` substitution.

    Single pass on purpose: substituted content (an issue body, a traceback) may itself
    contain ``{{...}}`` and must never be re-expanded.
    """
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


def render(template_id: str, variables: dict) -> Prompt:
    """Render the one template. This is the only way a prompt is ever produced.

    ``variables`` supplies every ``{{placeholder}}`` used by the templates; values are
    stringified, and empty/whitespace-only values render as :data:`EMPTY_VALUE` so all
    suites share one prompt shape. Extra keys are allowed (adapters may pass more than the
    template consumes) but are not recorded as substituted.

    Raises :class:`PromptError` on a wrong template id or a missing variable — both are
    contract violations, never something to paper over at run time.
    """
    if template_id != TEMPLATE_ID:
        raise PromptError(
            f"template id mismatch: asked for {template_id!r}, this harness ships {TEMPLATE_ID!r}"
        )

    system = system_text()
    user_tpl = user_template()

    needed = placeholders(system) | placeholders(user_tpl)
    missing = sorted(needed - set(variables))
    if missing:
        raise PromptError(f"missing prompt variable(s): {', '.join(missing)}")

    used: dict[str, str] = {}
    for name in sorted(needed):
        value = variables[name]
        text = "" if value is None else str(value)
        used[name] = text if text.strip() else EMPTY_VALUE

    tool_schemas = tools()
    rendered_system = _substitute(system, used)
    rendered_user = _substitute(user_tpl, used)

    payload = {
        "template_id": TEMPLATE_ID,
        "system": rendered_system,
        "user": rendered_user,
        "tools": list(tool_schemas),
    }
    prompt_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()

    return Prompt(
        template_id=TEMPLATE_ID,
        system=rendered_system,
        user=rendered_user,
        tools=tool_schemas,
        prompt_sha256=prompt_sha256,
        variables=used,
    )


# --- CONTRACTS.md §2.4 directory digest -------------------------------------------------

_DIGEST_SKIP_DIRS = {".git", ".cache", "__pycache__"}
_DIGEST_SKIP_NAMES = {".DS_Store"}


def directory_digest(root: Path, *, allow_symlinks: bool = False) -> str:
    """Normative directory digest from CONTRACTS.md §2.4.

    Equivalent to ``sha256sum`` over every file, ``LC_ALL=C sort``-ed by relative path, so
    it can be reproduced by hand. Skips ``.git/``, ``.cache/``, ``__pycache__/``, ``*.pyc``
    and ``.DS_Store``. Symlinks are skipped, or raise when ``allow_symlinks`` is False and
    one is found (weights directories must not contain them).
    """
    root = Path(root).resolve()
    entries: list[tuple[bytes, str]] = []
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in _DIGEST_SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if path.is_symlink():
            if allow_symlinks:
                continue
            raise PromptError(f"symlink in digested directory: {path}")
        if not path.is_file():
            continue
        name = rel_parts[-1]
        if name in _DIGEST_SKIP_NAMES or name.endswith(".pyc") or name in _DIGEST_SKIP_DIRS:
            continue
        rel = "/".join(rel_parts)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append((rel.encode("utf-8"), digest.hexdigest()))

    entries.sort(key=lambda pair: pair[0])
    stream = b"".join(f"{h}  ".encode("ascii") + rel + b"\n" for rel, h in entries)
    return hashlib.sha256(stream).hexdigest()


def dir_sha256() -> str:
    """``prompt_dir_sha256`` for the manifest: the digest of this directory."""
    return directory_digest(PROMPTS_DIR, allow_symlinks=True)


def info() -> dict:
    """Everything ``run.sh`` needs to fill the manifest's prompt fields."""
    return {
        "template_id": TEMPLATE_ID,
        "prompt_dir": str(PROMPTS_DIR),
        "prompt_dir_sha256": dir_sha256(),
        "files": sorted(p.name for p in PROMPTS_DIR.iterdir() if p.is_file()),
        "variables": sorted(placeholders(system_text()) | placeholders(user_template())),
        "tools": [t.get("function", {}).get("name") for t in tools()],
    }
