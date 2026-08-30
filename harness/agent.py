#!/usr/bin/env python3
"""harness/agent.py — THE fixed agent loop for AgentTask AI-P153, "The Harness Variable".

The contribution of this study is that the agent is a *constant*. Iteration budget, retry
policy, sampling parameters, tool set, prompt template, compaction strategy and termination
rules are all defined once — in `harness/agent_config.json` and `harness/prompts/` — and are
byte-identical for every model. There is no `if model == ...` in this file and there never
may be: a per-model branch here silently turns the experiment into a comparison of harnesses.

Everything the manifest needs to prove that is exposed by `inference_params()`, which returns
exactly the `inference` block of CONTRACTS.md §2.1 for `run.sh` to embed.

Contract: docs/CONTRACTS.md §1 (driver/CLI), §2.1 (`inference`), §3 (raw-result/v1), §4
(closed failure taxonomy), §5 (adapters), §5.2 (harness-constant prompt invariant).

Subcommands
-----------
  run              execute attempts for one suite and append raw-result/v1 records
  inference-params emit the manifest `inference` block as canonical JSON
  prompt-info      emit template id, prompt_dir_sha256, tool names, template variables
  adapter-info     emit adapter path/version/sha256/consent for the manifest
  prompt-preview   render the first task's prompt (for `run.sh --dry-run`)

How `run.sh` calls it (the real path)::

    python3 -m harness.agent run --manifest <run_dir>/run-manifest.json \
        --run-dir <run_dir> --pass-idx <n> --summary-out <f> [--only-instances <file>]

With `--manifest`, the manifest is the authority for model, suite, endpoint, instance list
and every inference knob — the loop executes exactly what the published record claims it
did. The bare `--model/--suite/...` form exists for local debugging.

Exit codes mirror CONTRACTS.md §1.3 so `run.sh` can propagate them unchanged:
  0 ok · 1 usage · 2 config · 3 preflight · 4 incomplete · 5 grading-degraded · 130 interrupt

stdout is machine-readable only. `run` writes nothing to stdout; its summary goes to
`--summary-out` (a JSON file) or, failing that, to stderr as one `==> summary {...}` line.
Dependencies: Python 3.11 standard library only (urllib, not the openai client) plus the
suite adapters. Do not add third-party imports to this file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import prompts as prompt_pkg  # noqa: E402  (needs REPO_ROOT on sys.path)

CONFIG_PATH = HARNESS_DIR / "agent_config.json"

# --------------------------------------------------------------------------------------
# CONTRACTS.md §4 — the closed failure taxonomy. Adding a value is a raw-result major bump.
# This tuple is the single source of truth for `error_code` across the harness.
# --------------------------------------------------------------------------------------

ERROR_CODES = (
    "OK",
    # model / agent — counted in the denominator
    "NO_PATCH",
    "PATCH_MALFORMED",
    "TESTS_FAIL",
    "TESTS_REGRESSION",
    "BUDGET_ITERATIONS",
    "BUDGET_TOKENS",
    "BUDGET_WALLCLOCK",
    "MODEL_CONTEXT_OVERFLOW",
    "MODEL_MALFORMED_TOOL_CALL",
    "MODEL_LOOP",
    "MODEL_REFUSAL",
    "MODEL_EMPTY_RESPONSE",
    # serving — counted in the denominator
    "SERVER_ERROR",
    "SERVER_UNAVAILABLE",
    # infrastructure — excluded from the denominator
    "INFRA_SANDBOX",
    "INFRA_GRADER",
    "INFRA_HOST",
    "INFRA_UNKNOWN",
)
ERROR_CODE_SET = frozenset(ERROR_CODES)
# NOTE FOR REVIEW: CONTRACTS.md §4 says "exactly these 18 values" but its four tables
# enumerate 19 (1 success + 12 model/agent + 2 serving + 4 infrastructure). The enumerated
# tables are authoritative and are what is implemented here; the prose count is off by one.
assert len(ERROR_CODE_SET) == len(ERROR_CODES), "duplicate error code in the taxonomy"

SUITES = ("swebench-verified", "swebench-pro", "agenttask")
SUITE_MODULES = {
    "swebench-verified": "swebench_verified",
    "swebench-pro": "swebench_pro",
    "agenttask": "agenttask",
}
DEFAULT_SEED_FILES = {
    "swebench-verified": "suites/verified-100.json",
    "swebench-pro": "suites/pro-50.json",
    "agenttask": "suites/agenttask/seed.json",
}
DEFAULT_ENDPOINT = "http://localhost:8000/v1"

#: Conservative refusal markers. Only consulted when the model stopped *without* calling a
#: tool and left no patch — otherwise a task about, say, a security parser would be
#: misclassified. Kept here (not in the prompt dir) because this is classification logic.
REFUSAL_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bI (?:can(?:no|')t|will not|won't|am unable to) (?:help|assist|comply|do that|complete)",
        r"\bI'm (?:sorry|afraid)[,.]? (?:but )?I (?:can(?:no|')t|won't|am unable)",
        r"\bas an AI\b.{0,60}\b(?:can(?:no|')t|unable)",
        r"\bI must (?:decline|refuse)\b",
    )
)

CONTEXT_OVERFLOW_MARKERS = (
    "maximum context length",
    "longer than the maximum",
    "context length",
    "max_model_len",
    "reduce the length of the messages",
    "please reduce the length",
)


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    """ISO-8601 UTC with a trailing Z and second resolution, as used everywhere on disk."""
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(obj: Any) -> bytes:
    return prompt_pkg.canonical_json(obj)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_line(obj: Any) -> str:
    """CONTRACTS.md §0 JSONL form: compact, sorted keys, one line, newline-terminated."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def truncate(text: str, limit: int, marker: str = "") -> tuple[str, int]:
    """Truncate to `limit` bytes of UTF-8, returning (text, bytes_dropped)."""
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, 0
    kept = raw[:limit].decode("utf-8", "ignore")
    return kept + marker, len(raw) - limit


def _p50(values: list[float]) -> int | None:
    return int(statistics.median(values)) if values else None


def _pmax(values: list[float]) -> int | None:
    return int(max(values)) if values else None


def stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# Configuration — the control variable
# --------------------------------------------------------------------------------------

#: Compiled-in defaults, identical to the committed agent_config.json. They exist so the
#: agent still runs (and still runs *the same way*) if the file is missing, and so a drifted
#: file is detectable rather than silently authoritative.
DEFAULT_CONFIG: dict[str, Any] = {
    "schema": "agent-config/v1",
    "template_id": "agent-v1",
    "inference": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 20260830,
        "max_tokens": 8192,
        "stop": [],
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_iters": 40,
        "max_attempt_tokens": 600000,
        "task_timeout_s": 1800,
        "concurrency": 4,
        "passes": 3,
        "retry_policy": {
            "max_retries": 3,
            "retry_on": ["http_5xx", "connection_error", "empty_response", "malformed_tool_call"],
            "backoff": "exponential",
            "base_delay_s": 2,
            "max_delay_s": 30,
            "jitter": False,
            "retries_count_against_iteration_budget": False,
        },
    },
    "loop": {
        "no_tool_call_limit": 3,
        "loop_repeat_threshold": 5,
        "budget_low_warning_at": 5,
        "max_tool_calls_per_turn": 4,
        "compaction": {"keep_recent_tool_results": 6, "max_compactions": 2, "elide_over_bytes": 512},
        "tools": {
            "command_timeout_default_s": 120,
            "command_timeout_max_s": 600,
            "tool_output_max_bytes": 16384,
            "file_read_max_lines": 400,
            "search_max_results": 100,
            "create_file_max_bytes": 262144,
        },
        "trajectory_content_max_bytes": 8192,
        "http_timeout_s": 900,
        "health_check_timeout_s": 15,
        "send_vllm_sampling_extras": True,
    },
}

#: The manifest `inference` block, in CONTRACTS.md §2.1 order. Any key here that is not in
#: this tuple (or vice versa) is a contract violation and fails fast.
INFERENCE_KEYS = (
    "endpoint",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "max_iters",
    "max_attempt_tokens",
    "task_timeout_s",
    "concurrency",
    "passes",
    "retry_policy",
)


class ConfigError(Exception):
    """Bad configuration. Maps to exit code 2."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Path | None = None) -> dict:
    """Load the held-constant knobs. `agent_config.json` is authoritative when present."""
    path = Path(path) if path else CONFIG_PATH
    if not path.exists():
        stderr(f"==> warning: {path} missing — using compiled-in defaults (nonconformant)")
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read agent config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    data.pop("_comment", None)
    cfg = _deep_merge(DEFAULT_CONFIG, data)
    unknown = set(cfg["inference"]) - set(DEFAULT_CONFIG["inference"])
    if unknown:
        raise ConfigError(f"{path}: unknown inference key(s) {sorted(unknown)}")
    if cfg.get("template_id") != prompt_pkg.TEMPLATE_ID:
        raise ConfigError(
            f"{path} pins template_id {cfg.get('template_id')!r} but harness/prompts/ ships "
            f"{prompt_pkg.TEMPLATE_ID!r} — the control variable is inconsistent"
        )
    return cfg


def inference_params(
    cfg: dict,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    max_iters: int | None = None,
    task_timeout_s: int | None = None,
    concurrency: int | None = None,
    passes: int | None = None,
) -> dict:
    """Return exactly the manifest `inference` block (CONTRACTS.md §2.1) for run.sh.

    This is the function the manifest is built from: whatever it says is what the loop
    actually does, because the loop reads the same dict.
    """
    params = copy.deepcopy(cfg["inference"])
    params["endpoint"] = endpoint
    for key, value in (
        ("max_iters", max_iters),
        ("task_timeout_s", task_timeout_s),
        ("concurrency", concurrency),
        ("passes", passes),
    ):
        if value is not None:
            params[key] = value
    for key in ("max_iters", "max_attempt_tokens", "task_timeout_s", "concurrency", "passes"):
        if not isinstance(params[key], int) or params[key] < 1:
            raise ConfigError(f"inference.{key} must be a positive integer, got {params[key]!r}")
    missing = [k for k in INFERENCE_KEYS if k not in params]
    extra = [k for k in params if k not in INFERENCE_KEYS]
    if missing or extra:
        raise ConfigError(f"inference block mismatch: missing={missing} extra={extra}")
    return {key: params[key] for key in INFERENCE_KEYS}


def nonconformant_reasons(cfg: dict, params: dict) -> list[str]:
    """Which held-constant knobs deviate from the compiled study defaults, if any."""
    reasons = []
    defaults = DEFAULT_CONFIG["inference"]
    for key, default in defaults.items():
        if key in ("concurrency", "passes"):  # throughput knobs, not verdict knobs
            continue
        if params.get(key) != default:
            reasons.append(f"inference.{key}={params.get(key)!r} != study default {default!r}")
    return reasons


# --------------------------------------------------------------------------------------
# Attempt-level control flow
# --------------------------------------------------------------------------------------


class AttemptAbort(Exception):
    """Terminates one attempt with an explicit taxonomy code (CONTRACTS.md §4)."""

    def __init__(self, code: str, detail: str = "") -> None:
        if code not in ERROR_CODE_SET:
            raise ValueError(f"{code!r} is not in the closed taxonomy")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail[:512]


class RetryableLLMError(Exception):
    """A completion failure the retry policy is allowed to retry."""

    def __init__(self, reason: str, code: str, detail: str, notice_key: str | None = None) -> None:
        super().__init__(detail)
        self.reason = reason  # one of retry_policy.retry_on
        self.code = code  # taxonomy code if retries are exhausted
        self.detail = detail
        self.notice_key = notice_key  # notice injected before re-requesting, if any


class ContextOverflow(Exception):
    """The server rejected the request for length; the caller compacts and retries once."""


_STOP = threading.Event()
_STOP_SIGNAL: list[int] = []


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        if not _STOP_SIGNAL:
            _STOP_SIGNAL.append(signum)
        _STOP.set()
        stderr(f"==> signal {signal.Signals(signum).name} — draining in-flight attempts")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except ValueError:  # not on the main thread
            pass


def _check_stop() -> None:
    if _STOP.is_set():
        raise AttemptAbort("INFRA_HOST", "harness received SIGINT/SIGTERM")


# --------------------------------------------------------------------------------------
# OpenAI-compatible client — stdlib only, on purpose (tiny dependency surface)
# --------------------------------------------------------------------------------------


class LLMClient:
    """Minimal chat-completions client for vLLM's OpenAI-compatible server.

    Non-streaming by design: streaming buys per-token latency (TTFT) at the cost of partial
    -response failure modes, and the headline metric of this study is cost per resolved
    task, not TTFT. `latency_ms.ttft_*` is therefore recorded as null (CONTRACTS.md §3).
    """

    def __init__(self, endpoint: str, model: str, cfg: dict) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.inf = cfg["inference"]
        self.loop_cfg = cfg["loop"]
        self.api_key = os.environ.get("HARNESS_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    # -- transport ---------------------------------------------------------------------

    def _request(self, path: str, payload: dict | None, timeout: float) -> tuple[int, bytes]:
        url = f"{self.endpoint}{path}"
        data = canonical_json(payload) if payload is not None else None
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed host)
            return resp.status, resp.read()

    def health(self) -> list[str]:
        """Return the served model ids. Raises on an unreachable/unhealthy endpoint."""
        _, body = self._request("/models", None, float(self.loop_cfg["health_check_timeout_s"]))
        data = json.loads(body.decode("utf-8"))
        return [entry.get("id", "") for entry in data.get("data", [])]

    def is_up(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    # -- completions -------------------------------------------------------------------

    def _payload(self, messages: list[dict], tools: Iterable[dict]) -> dict:
        inf = self.inf
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": list(tools),
            "tool_choice": "auto",
            "temperature": inf["temperature"],
            "top_p": inf["top_p"],
            "seed": inf["seed"],
            "max_tokens": inf["max_tokens"],
            "presence_penalty": inf["presence_penalty"],
            "frequency_penalty": inf["frequency_penalty"],
            "stream": False,
        }
        if inf["stop"]:
            payload["stop"] = list(inf["stop"])
        if self.loop_cfg.get("send_vllm_sampling_extras", True):
            # vLLM honours these as top-level extras; a strict OpenAI server would 400.
            payload["top_k"] = inf["top_k"]
            payload["repetition_penalty"] = inf["repetition_penalty"]
        return payload

    def complete(self, messages: list[dict], tools: Iterable[dict]) -> tuple[dict, float]:
        """One completion request. Raises RetryableLLMError / ContextOverflow / AttemptAbort."""
        payload = self._payload(messages, tools)
        started = time.monotonic()
        try:
            status, body = self._request(
                "/chat/completions", payload, float(self.loop_cfg["http_timeout_s"])
            )
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:512]
            except Exception:
                pass
            if exc.code >= 500:
                raise RetryableLLMError("http_5xx", "SERVER_ERROR", f"HTTP {exc.code}: {detail}")
            if any(marker in detail.lower() for marker in CONTEXT_OVERFLOW_MARKERS):
                raise ContextOverflow(detail)
            raise AttemptAbort("SERVER_ERROR", f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            raise RetryableLLMError(
                "connection_error", "SERVER_UNAVAILABLE", f"{type(exc).__name__}: {exc}"
            )
        elapsed_ms = (time.monotonic() - started) * 1000.0

        try:
            resp = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetryableLLMError("http_5xx", "SERVER_ERROR", f"unparseable response body: {exc}")
        if status >= 400 or "choices" not in resp:
            message = json.dumps(resp.get("error", resp))[:512]
            if any(marker in message.lower() for marker in CONTEXT_OVERFLOW_MARKERS):
                raise ContextOverflow(message)
            raise RetryableLLMError("http_5xx", "SERVER_ERROR", f"HTTP {status}: {message}")
        if not resp["choices"]:
            raise RetryableLLMError(
                "empty_response", "MODEL_EMPTY_RESPONSE", "no choices in response", "empty_response"
            )
        return resp, elapsed_ms

    def backoff_delay(self, attempt_index: int) -> float:
        """Exponential backoff from retry_policy; `jitter: false` keeps runs reproducible."""
        policy = self.inf["retry_policy"]
        delay = float(policy["base_delay_s"]) * (2 ** attempt_index)
        return min(delay, float(policy["max_delay_s"]))


# --------------------------------------------------------------------------------------
# Execution environments
#
# The agent needs a workspace: a checkout of the project at `task.base_commit` in which
# commands run and files are edited. Two backends, chosen from `task.environment`:
#
#   DockerExecutor  when environment["image"] is set and docker is available — the SWE-bench
#                   convention, repo already present at environment["workdir"] (/testbed).
#   LocalExecutor   when environment["workspace"] (a directory to clone) or a repo cache is
#                   available — used for agenttask and for local development.
#
# Both are deliberately thin: one `run()` plus file read/write. Every failure to build a
# workspace is INFRA_SANDBOX and is excluded from the denominator (CONTRACTS.md §4).
# --------------------------------------------------------------------------------------


class SandboxError(Exception):
    """Workspace could not be built or used. Maps to INFRA_SANDBOX."""


class Executor:
    """Command + file interface to one task workspace."""

    workdir: str = "/"

    def run(self, command: str, timeout_s: int) -> tuple[int, str]:
        raise NotImplementedError

    def read_text(self, rel_path: str) -> str:
        raise NotImplementedError

    def write_text(self, rel_path: str, content: str) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        return None


def safe_rel(path: str) -> str:
    """Reject absolute paths and `..` escapes: tool paths are repo-root relative."""
    cleaned = (path or "").strip()
    if not cleaned:
        raise SandboxError("empty path")
    if cleaned.startswith("/") or cleaned.startswith("~"):
        raise SandboxError(f"path must be relative to the repository root: {path!r}")
    parts = [p for p in Path(cleaned).parts if p not in (".",)]
    if any(p == ".." for p in parts):
        raise SandboxError(f"path escapes the workspace: {path!r}")
    return "/".join(parts)


class LocalExecutor(Executor):
    """Runs in a directory on this host."""

    def __init__(self, workdir: Path) -> None:
        self.root = Path(workdir).resolve()
        self.workdir = str(self.root)

    def run(self, command: str, timeout_s: int) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.output or b"").decode("utf-8", "replace")
            return 124, partial + f"\n[harness] command exceeded {timeout_s}s and was killed"
        return proc.returncode, proc.stdout.decode("utf-8", "replace")

    def _abs(self, rel_path: str) -> Path:
        return self.root / safe_rel(rel_path)

    def read_text(self, rel_path: str) -> str:
        try:
            return self._abs(rel_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SandboxError(str(exc)) from exc

    def write_text(self, rel_path: str, content: str) -> None:
        target = self._abs(rel_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(str(exc)) from exc


class DockerExecutor(Executor):
    """Runs inside a per-attempt container built from the task's image."""

    def __init__(self, image: str, workdir: str, container_name: str) -> None:
        self.image = image
        self.workdir = workdir or "/testbed"
        self.name = container_name
        self._started = False

    def start(self) -> None:
        extra = os.environ.get("HARNESS_DOCKER_ARGS", "")
        cmd = ["docker", "run", "-d", "--name", self.name, "--entrypoint", "/bin/sh"]
        if extra:
            cmd.extend(extra.split())
        cmd.extend([self.image, "-c", "sleep infinity"])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=900)
        if proc.returncode != 0:
            raise SandboxError(f"docker run failed: {proc.stderr.strip()[:300]}")
        self._started = True

    def _exec(self, argv: list[str], timeout_s: int, stdin_text: str | None = None):
        cmd = ["docker", "exec", "-w", self.workdir]
        if stdin_text is not None:
            cmd.append("-i")
        cmd.append(self.name)
        cmd.extend(argv)
        return subprocess.run(
            cmd,
            input=stdin_text.encode("utf-8") if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )

    def run(self, command: str, timeout_s: int) -> tuple[int, str]:
        try:
            proc = self._exec(["/bin/sh", "-c", command], timeout_s)
        except subprocess.TimeoutExpired as exc:
            partial = (exc.output or b"").decode("utf-8", "replace")
            return 124, partial + f"\n[harness] command exceeded {timeout_s}s and was killed"
        return proc.returncode, proc.stdout.decode("utf-8", "replace")

    def read_text(self, rel_path: str) -> str:
        rel = safe_rel(rel_path)
        proc = self._exec(["/bin/cat", "--", rel], 120)
        out = proc.stdout.decode("utf-8", "replace")
        if proc.returncode != 0:
            raise SandboxError(out.strip()[:300])
        return out

    def write_text(self, rel_path: str, content: str) -> None:
        rel = safe_rel(rel_path)
        parent = str(Path(rel).parent)
        script = f'mkdir -p -- "{parent}" && cat > "{rel}"'
        proc = self._exec(["/bin/sh", "-c", script], 300, stdin_text=content)
        if proc.returncode != 0:
            raise SandboxError(proc.stdout.decode("utf-8", "replace").strip()[:300])

    def cleanup(self) -> None:
        if self._started:
            subprocess.run(
                ["docker", "rm", "-f", self.name],
                capture_output=True,
                check=False,
                timeout=300,
            )
            self._started = False


def _adapter_for(task: Any) -> Any:
    """Resolve the adapter module that produced `task`, or None if unavailable."""
    try:
        from harness import adapters  # local import: keeps agent.py importable standalone

        return adapters.get(getattr(task, "suite", ""))
    except Exception:  # noqa: BLE001 — a missing adapter just means no materialize() hook
        return None


def build_workspace(task: Any, attempt_slug: str, scratch_root: Path) -> Executor:
    """Create the workspace for one attempt. Every failure here is INFRA_SANDBOX."""
    env = dict(getattr(task, "environment", {}) or {})
    image = (env.get("image") or "").strip()
    workdir = (env.get("workdir") or "/testbed").strip()
    base_commit = (getattr(task, "base_commit", "") or "").strip()

    executor: Executor
    if image and os.environ.get("HARNESS_FORCE_LOCAL", "") != "1":
        if not _have("docker"):
            raise SandboxError(f"task requires image {image} but docker is not available")
        docker = DockerExecutor(image, workdir, f"harness-{attempt_slug}"[:60])
        docker.start()
        executor = docker
    else:
        source = env.get("workspace") or env.get("repo_path") or ""
        if not source:
            cache = os.environ.get("HARNESS_REPO_CACHE", "")
            repo = (getattr(task, "repo", "") or "").replace("/", "__")
            if cache and repo and (Path(cache) / repo).exists():
                source = str(Path(cache) / repo)
        if not source:
            # A suite may ship its own workspace (agenttask carries a repo snapshot in its
            # data pack rather than a docker image). Adapters expose that as an optional
            # `materialize(task, dest)` hook; hidden tests are never included here — grade()
            # overlays them on its own copy.
            adapter = _adapter_for(task)
            materialize = getattr(adapter, "materialize", None)
            if callable(materialize):
                dest = scratch_root / attempt_slug
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    materialize(task, dest, include_hidden_tests=False)
                except Exception as exc:  # noqa: BLE001 — any failure here is INFRA_SANDBOX
                    raise SandboxError(
                        f"adapter materialize() failed for {task.instance_id}: {exc}"
                    ) from exc
                return LocalExecutor(dest)
        if not source or not Path(source).exists():
            raise SandboxError(
                "no workspace: task has no image, no environment['workspace'], no adapter "
                "materialize() hook, and no HARNESS_REPO_CACHE entry"
            )
        dest = scratch_root / attempt_slug
        dest.parent.mkdir(parents=True, exist_ok=True)
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(dest)],
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        if clone.returncode != 0:
            raise SandboxError(f"git clone failed: {clone.stderr.strip()[:300]}")
        executor = LocalExecutor(dest)

    try:
        if base_commit:
            rc, out = executor.run(f"git checkout --quiet --force {base_commit}", 600)
            if rc != 0:
                raise SandboxError(f"git checkout {base_commit} failed: {out.strip()[:300]}")
        rc, _ = executor.run("git rev-parse --is-inside-work-tree", 60)
        if rc != 0:
            raise SandboxError("workspace is not a git work tree — no diff can be produced")
        for cmd in env.get("setup_cmds") or []:
            rc, out = executor.run(str(cmd), 3600)
            if rc != 0:
                raise SandboxError(f"setup command failed ({cmd}): {out.strip()[-300:]}")
    except SandboxError:
        executor.cleanup()
        raise
    except Exception as exc:  # noqa: BLE001 — any prep failure is a sandbox failure
        executor.cleanup()
        raise SandboxError(f"{type(exc).__name__}: {exc}") from exc
    return executor


def _have(binary: str) -> bool:
    from shutil import which

    return which(binary) is not None


def capture_patch(executor: Executor, max_bytes: int = 1_000_000) -> str:
    """The attempt's diff: everything in the work tree against the base commit."""
    executor.run("git add -A -- .", 300)
    rc, out = executor.run("git diff --cached --no-color --no-ext-diff", 600)
    if rc != 0:
        return ""
    if len(out.encode("utf-8")) > max_bytes:
        # A diff this large is never a real patch; record it truncated rather than blowing
        # up the bundle. It will fail to apply and be graded PATCH_MALFORMED.
        out, _ = truncate(out, max_bytes, "\n[harness] diff truncated\n")
    return out


def patch_stats(patch: str) -> tuple[int, int, int]:
    """(files_changed, lines_added, lines_removed) from a unified diff."""
    files = added = removed = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            files += 1
        elif line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return files, added, removed


# --------------------------------------------------------------------------------------
# Tools — the fixed tool set from harness/prompts/tools.json, implemented once
# --------------------------------------------------------------------------------------


class ToolError(Exception):
    """An ordinary tool failure. Reported back to the model; costs one iteration."""


class MalformedToolCall(Exception):
    """Unparseable arguments or an unknown tool — retried per the retry policy (§4)."""


class Toolbox:
    """Implements every tool in harness/prompts/tools.json against one workspace."""

    def __init__(self, executor: Executor, task: Any, cfg: dict) -> None:
        self.executor = executor
        self.task = task
        self.tool_cfg = cfg["loop"]["tools"]
        self.submitted: str | None = None

    #: Must stay in lock-step with harness/prompts/tools.json — verified below.
    NAMES = frozenset(
        {"run_command", "read_file", "search_files", "edit_file", "create_file", "run_tests", "submit"}
    )

    @property
    def names(self) -> frozenset[str]:
        return self.NAMES

    def dispatch(self, name: str, args: dict) -> str:
        if name not in self.names:
            raise MalformedToolCall(f"unknown tool {name!r}; available: {sorted(self.names)}")
        handler: Callable[[dict], str] = getattr(self, f"_tool_{name}")
        try:
            return handler(args)
        except (MalformedToolCall, ToolError):
            raise
        except SandboxError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — a tool bug must not kill the attempt
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    # -- helpers -----------------------------------------------------------------------

    def _timeout(self, args: dict) -> int:
        requested = args.get("timeout_s")
        default = int(self.tool_cfg["command_timeout_default_s"])
        maximum = int(self.tool_cfg["command_timeout_max_s"])
        if requested is None:
            return default
        try:
            return max(1, min(int(requested), maximum))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _require(args: dict, key: str, kind: type) -> Any:
        if key not in args:
            raise MalformedToolCall(f"missing required argument {key!r}")
        value = args[key]
        if not isinstance(value, kind):
            raise MalformedToolCall(
                f"argument {key!r} must be {kind.__name__}, got {type(value).__name__}"
            )
        return value

    def _format_run(self, rc: int, out: str) -> str:
        return f"exit_code: {rc}\n--- output ---\n{out}" if out.strip() else f"exit_code: {rc}\n(no output)"

    # -- tools -------------------------------------------------------------------------

    def _tool_run_command(self, args: dict) -> str:
        command = self._require(args, "command", str)
        rc, out = self.executor.run(command, self._timeout(args))
        return self._format_run(rc, out)

    def _tool_read_file(self, args: dict) -> str:
        path = self._require(args, "path", str)
        text = self.executor.read_text(path)
        lines = text.splitlines()
        max_lines = int(self.tool_cfg["file_read_max_lines"])
        start = int(args.get("start_line") or 1)
        start = max(1, start)
        end = args.get("end_line")
        end = int(end) if end else start + max_lines - 1
        end = min(end, len(lines), start + max_lines - 1)
        if start > len(lines):
            raise ToolError(f"{path} has {len(lines)} lines; start_line {start} is past the end")
        body = "\n".join(f"{n:6d}\t{lines[n - 1]}" for n in range(start, end + 1))
        header = f"{path} (lines {start}-{end} of {len(lines)})"
        if end < len(lines):
            body += f"\n[harness] {len(lines) - end} more line(s); call read_file again with start_line={end + 1}"
        return f"{header}\n{body}"

    def _tool_search_files(self, args: dict) -> str:
        pattern = self._require(args, "pattern", str)
        path = args.get("path") or "."
        glob = args.get("glob")
        limit = int(args.get("max_results") or self.tool_cfg["search_max_results"])
        limit = max(1, min(limit, int(self.tool_cfg["search_max_results"])))
        include = f" --include={_shquote(str(glob))}" if glob else ""
        cmd = (
            f"grep -rnIE{include} -e {_shquote(pattern)} -- {_shquote(safe_rel(path)) if path != '.' else '.'}"
            f" 2>/dev/null | head -n {limit}"
        )
        rc, out = self.executor.run(cmd, self._timeout(args))
        if not out.strip():
            return f"no matches for {pattern!r} under {path}"
        count = len(out.strip().splitlines())
        suffix = f"\n[harness] showing first {limit} matches; narrow the pattern" if count >= limit else ""
        return out + suffix

    def _tool_edit_file(self, args: dict) -> str:
        path = self._require(args, "path", str)
        old = self._require(args, "old_str", str)
        new = self._require(args, "new_str", str)
        replace_all = bool(args.get("replace_all", False))
        if old == "":
            raise ToolError("old_str must not be empty; use create_file to write a whole file")
        text = self.executor.read_text(path)
        occurrences = text.count(old)
        if occurrences == 0:
            raise ToolError(
                f"old_str not found in {path}. Re-read the file and copy the target text "
                f"verbatim, including indentation."
            )
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"old_str occurs {occurrences} times in {path}. Include more surrounding "
                f"lines to make it unique, or set replace_all."
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        self.executor.write_text(path, updated)
        changed = occurrences if replace_all else 1
        return f"edited {path}: {changed} replacement(s)"

    def _tool_create_file(self, args: dict) -> str:
        path = self._require(args, "path", str)
        content = self._require(args, "content", str)
        limit = int(self.tool_cfg["create_file_max_bytes"])
        if len(content.encode("utf-8")) > limit:
            raise ToolError(f"content exceeds the {limit}-byte limit for create_file")
        self.executor.write_text(path, content)
        return f"wrote {path} ({len(content.encode('utf-8'))} bytes, {len(content.splitlines())} lines)"

    def _tool_run_tests(self, args: dict) -> str:
        env = dict(getattr(self.task, "environment", {}) or {})
        test_cmd = (env.get("test_cmd") or "").strip()
        if not test_cmd:
            raise ToolError("this task defines no test command; use run_command instead")
        node_ids = args.get("node_ids") or []
        if node_ids and not isinstance(node_ids, list):
            raise MalformedToolCall("node_ids must be an array of strings")
        selection = " ".join(_shquote(str(n)) for n in node_ids)
        command = f"{test_cmd} {selection}".strip()
        # Test runs get the long ceiling by default: a suite that needs 4 minutes must not
        # be scored as a failure because of the ordinary command timeout.
        timeout = self._timeout(args) if args.get("timeout_s") else int(self.tool_cfg["command_timeout_max_s"])
        rc, out = self.executor.run(command, timeout)
        return f"$ {command}\n" + self._format_run(rc, out)

    def _tool_submit(self, args: dict) -> str:
        self.submitted = str(args.get("summary", ""))[:2000]
        return "submitted"


def _shquote(text: str) -> str:
    """POSIX single-quote quoting (shlex.quote, inlined to keep the import list short)."""
    return "'" + text.replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------------------
# Trajectory writing (CONTRACTS.md §3.2)
# --------------------------------------------------------------------------------------


class TrajectoryWriter:
    """One JSONL file per attempt; `content` is truncated to a fixed byte budget."""

    def __init__(self, path: Path, max_content_bytes: int) -> None:
        self.path = path
        self.max_content_bytes = max_content_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.records = 0

    def write(self, *, role: str, kind: str, content: str = "", **extra: Any) -> None:
        text, dropped = truncate(content or "", self.max_content_bytes)
        record: dict[str, Any] = {
            "i": self.records,
            "t": iso(utcnow()),
            "role": role,
            "kind": kind,
            "content": text,
        }
        if dropped:
            record["content_truncated_bytes"] = dropped
        record.update({k: v for k, v in extra.items() if v is not None})
        self._handle.write(jsonl_line(record))
        self._handle.flush()
        self.records += 1

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


class Stats:
    """Per-attempt counters that end up in the raw-result record (CONTRACTS.md §3)."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0
        self.llm_calls = 0
        self.iterations = 0
        self.tool_calls = 0
        self.retries = 0
        self.latencies_ms: list[float] = []

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add_usage(self, usage: dict) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        self.cached_prompt_tokens += int(details.get("cached_tokens") or 0)


class RunContext:
    """Everything one attempt needs; identical for every attempt in the run."""

    def __init__(
        self,
        *,
        cfg: dict,
        params: dict,
        client: LLMClient,
        adapter: Any,
        run_id: str,
        run_dir: Path,
        model: str,
        suite: str,
        consent_class: str,
        scratch_root: Path,
        partitions: dict[str, str] | None = None,
        cents_per_hour: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.params = params
        self.client = client
        self.adapter = adapter
        self.run_id = run_id
        self.run_dir = run_dir
        self.model = model
        self.suite = suite
        self.consent_class = consent_class
        self.scratch_root = scratch_root
        self.partitions = partitions or {}
        self.cents_per_hour = cents_per_hour
        self.notices = prompt_pkg.notices()

    def partition_of(self, task: Any) -> str:
        """CONTRACTS.md §3.1: resolved from partitions.json, `unpartitioned` if absent."""
        qualified = getattr(task, "qualified_id", "") or f"{self.suite}::{task.instance_id}"
        return self.partitions.get(qualified) or getattr(task, "partition", "") or "unpartitioned"


def compact_messages(messages: list[dict], cfg: dict, notices: dict) -> bool:
    """The harness's ONE fixed compaction step (CONTRACTS.md §4 MODEL_CONTEXT_OVERFLOW).

    Elides the content of older tool results, keeping the most recent ones intact. Fixed
    and identical for every model — no adaptive summarisation, which would be a per-model
    confound.
    """
    conf = cfg["loop"]["compaction"]
    keep = int(conf["keep_recent_tool_results"])
    over = int(conf["elide_over_bytes"])
    marker = notices["compaction"]
    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    victims = tool_indexes[:-keep] if keep > 0 else tool_indexes
    changed = False
    for index in victims:
        content = messages[index].get("content") or ""
        if content != marker and len(content.encode("utf-8")) > over:
            messages[index]["content"] = marker
            changed = True
    return changed


def parse_tool_calls(raw_calls: list, tool_names: frozenset[str]) -> list[dict]:
    """Validate the model's tool calls. Anything unparseable is MalformedToolCall (§4)."""
    parsed = []
    for call in raw_calls:
        if not isinstance(call, dict):
            raise MalformedToolCall("tool call is not an object")
        function = call.get("function") or {}
        name = function.get("name")
        if name not in tool_names:
            raise MalformedToolCall(f"unknown tool {name!r}; available: {sorted(tool_names)}")
        raw_args = function.get("arguments", "{}")
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args or "{}")
            except (TypeError, json.JSONDecodeError) as exc:
                raise MalformedToolCall(f"arguments for {name} are not valid JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise MalformedToolCall(f"arguments for {name} must be a JSON object")
        parsed.append(
            {
                "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": name,
                "args": args,
                "args_sha256": sha256_text(json.dumps(args, sort_keys=True, separators=(",", ":"))),
                "args_bytes": len(json.dumps(args, separators=(",", ":")).encode("utf-8")),
                "raw": call,
            }
        )
    return parsed


def complete_with_retry(
    ctx: RunContext,
    messages: list[dict],
    tools: Iterable[dict],
    tool_names: frozenset[str],
    stats: Stats,
    traj: TrajectoryWriter,
) -> tuple[dict, list[dict], float]:
    """One agent turn's completion, including the whole retry policy.

    Retries do NOT consume an iteration (`retries_count_against_iteration_budget: false`)
    but they DO count in `llm_calls` and `tokens`, because they cost GPU time.
    """
    policy = ctx.params["retry_policy"]
    max_retries = int(policy["max_retries"])
    retry_on = set(policy["retry_on"])
    max_compactions = int(ctx.cfg["loop"]["compaction"]["max_compactions"])
    attempt = 0
    compactions = 0

    while True:
        _check_stop()
        stats.llm_calls += 1
        try:
            resp, elapsed_ms = ctx.client.complete(messages, tools)
            usage = resp.get("usage") or {}
            stats.add_usage(usage)
            stats.latencies_ms.append(elapsed_ms)

            choice = resp["choices"][0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            raw_calls = message.get("tool_calls") or []
            if not content and not raw_calls:
                raise RetryableLLMError(
                    "empty_response",
                    "MODEL_EMPTY_RESPONSE",
                    f"empty completion (finish_reason={choice.get('finish_reason')})",
                    "empty_response",
                )
            try:
                calls = parse_tool_calls(raw_calls, tool_names)
            except MalformedToolCall as exc:
                raise RetryableLLMError(
                    "malformed_tool_call", "MODEL_MALFORMED_TOOL_CALL", str(exc), "malformed_tool_call"
                ) from exc

            assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            if raw_calls:
                assistant["tool_calls"] = raw_calls
            return assistant, calls, elapsed_ms

        except ContextOverflow as exc:
            if compactions >= max_compactions or not compact_messages(messages, ctx.cfg, ctx.notices):
                raise AttemptAbort("MODEL_CONTEXT_OVERFLOW", str(exc)[:512])
            compactions += 1
            stats.retries += 1
            traj.write(
                role="system",
                kind="error",
                content=f"context overflow; applied compaction {compactions}/{max_compactions}",
            )
            continue

        except RetryableLLMError as exc:
            if exc.reason not in retry_on or attempt >= max_retries:
                raise AttemptAbort(exc.code, exc.detail[:512])
            attempt += 1
            stats.retries += 1
            traj.write(
                role="system",
                kind="error",
                content=f"retry {attempt}/{max_retries} after {exc.reason}: {exc.detail}",
            )
            if exc.notice_key:
                messages.append({"role": "user", "content": ctx.notices[exc.notice_key]})
            time.sleep(ctx.client.backoff_delay(attempt - 1))
            continue


def agent_loop(
    ctx: RunContext,
    prompt: Any,
    toolbox: Toolbox,
    stats: Stats,
    traj: TrajectoryWriter,
    log: Callable[[str], None],
    deadline: float,
    assistant_texts: list[str],
) -> tuple[str | None, str]:
    """Run the bounded edit/run/test loop. Returns (terminal_code_or_None, detail).

    `None` means the agent terminated on its own terms and the grader decides the outcome.
    `assistant_texts` is filled in place with the model's prose, for refusal detection.
    """
    cfg_loop = ctx.cfg["loop"]
    max_iters = int(ctx.params["max_iters"])
    max_tokens_total = int(ctx.params["max_attempt_tokens"])
    no_tool_call_limit = int(cfg_loop["no_tool_call_limit"])
    loop_threshold = int(cfg_loop["loop_repeat_threshold"])
    max_calls_per_turn = int(cfg_loop["max_tool_calls_per_turn"])
    warn_at = int(cfg_loop["budget_low_warning_at"])

    messages: list[dict] = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    traj.write(role="system", kind="prompt", content=prompt.system)
    traj.write(role="user", kind="prompt", content=prompt.user, prompt_sha256=prompt.prompt_sha256)

    consecutive_no_tool_calls = 0
    recent_calls: list[str] = []
    warned = False

    for iteration in range(max_iters):
        _check_stop()
        if time.monotonic() > deadline:
            return "BUDGET_WALLCLOCK", f"wall-clock limit of {ctx.params['task_timeout_s']}s reached"
        if stats.total_tokens >= max_tokens_total:
            return "BUDGET_TOKENS", f"token budget of {max_tokens_total} reached"
        remaining = max_iters - iteration
        if remaining <= warn_at and not warned:
            messages.append({"role": "user", "content": ctx.notices["budget_low"]})
            warned = True

        assistant, calls, elapsed_ms = complete_with_retry(
            ctx, messages, prompt.tools, toolbox.names, stats, traj
        )
        stats.iterations += 1
        messages.append(assistant)
        if assistant.get("content"):
            assistant_texts.append(assistant["content"])
        traj.write(
            role="assistant",
            kind="completion",
            content=assistant.get("content") or "",
            latency_ms=int(elapsed_ms),
            tool_call_count=len(calls) or None,
        )
        log(f"iter {stats.iterations}/{max_iters}: {len(calls)} tool call(s), {int(elapsed_ms)}ms")

        if not calls:
            consecutive_no_tool_calls += 1
            if consecutive_no_tool_calls >= no_tool_call_limit:
                return None, f"agent stopped calling tools ({consecutive_no_tool_calls} turns)"
            messages.append({"role": "user", "content": ctx.notices["no_tool_call"]})
            continue
        consecutive_no_tool_calls = 0

        for index, call in enumerate(calls):
            signature = f"{call['name']}:{call['args_sha256']}"
            recent_calls.append(signature)
            if (
                len(recent_calls) >= loop_threshold
                and len(set(recent_calls[-loop_threshold:])) == 1
            ):
                traj.write(role="system", kind="error", content=ctx.notices["loop_detected"])
                return "MODEL_LOOP", f"{call['name']} repeated {loop_threshold} times with identical arguments"

            traj.write(
                role="assistant",
                kind="tool_call",
                content=json.dumps(call["args"], sort_keys=True)[: cfg_loop["trajectory_content_max_bytes"]],
                tool=call["name"],
                args_sha256=call["args_sha256"],
                args_bytes=call["args_bytes"],
                tool_call_id=call["id"],
            )

            if index >= max_calls_per_turn:
                result = (
                    f"[harness] ignored: at most {max_calls_per_turn} tool calls are executed "
                    f"per turn. Re-issue this call on its own."
                )
                is_error = True
            else:
                stats.tool_calls += 1
                try:
                    result = toolbox.dispatch(call["name"], call["args"])
                    is_error = False
                except MalformedToolCall as exc:
                    result = f"[harness] malformed tool call: {exc}"
                    is_error = True
                except ToolError as exc:
                    result = f"[harness] tool error: {exc}"
                    is_error = True

            shown, dropped = truncate(
                result,
                int(cfg_loop["tools"]["tool_output_max_bytes"]),
                ctx.notices["tool_output_truncated"],
            )
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": shown})
            traj.write(
                role="tool",
                kind="tool_result",
                content=shown,
                tool=call["name"],
                tool_call_id=call["id"],
                is_error=is_error or None,
                content_truncated_bytes=dropped or None,
            )

        if toolbox.submitted is not None:
            return None, "agent called submit"

        if time.monotonic() > deadline:
            return "BUDGET_WALLCLOCK", f"wall-clock limit of {ctx.params['task_timeout_s']}s reached"

    return "BUDGET_ITERATIONS", f"iteration budget of {max_iters} exhausted"


def looks_like_refusal(texts: list[str]) -> bool:
    """Conservative: only consulted when the agent stopped with no patch and no tool use."""
    tail = "\n".join(texts[-3:])
    return any(pattern.search(tail) for pattern in REFUSAL_PATTERNS)


# --------------------------------------------------------------------------------------
# One attempt = one (instance_id, pass_idx) pair = one raw-result/v1 record
# --------------------------------------------------------------------------------------


def attempt_id(run_id: str, instance_id: str, pass_idx: int) -> str:
    """CONTRACTS.md §3.1: `<run_id 6-hex suffix>-<instance_id>-<pass_idx>`."""
    suffix = run_id.rsplit("__", 1)[-1] if "__" in run_id else run_id[-6:]
    return f"{suffix}-{instance_id}-{pass_idx}"


def _verdict_field(verdict: Any, name: str, default: Any = None) -> Any:
    if isinstance(verdict, dict):
        return verdict.get(name, default)
    return getattr(verdict, name, default)


def grade_patch(ctx: RunContext, task: Any, patch: str) -> tuple[Any, str | None]:
    """Call the adapter's grader. Any failure of the grader itself is INFRA_GRADER (§5.3)."""
    try:
        verdict = ctx.adapter.grade(task, patch)
    except Exception as exc:  # noqa: BLE001 — GraderError and bugs both mean INFRA_GRADER
        return None, f"{type(exc).__name__}: {exc}"[:512]
    if verdict is None:
        return None, "adapter.grade returned None"
    return verdict, None


def run_attempt(ctx: RunContext, task: Any, pass_idx: int) -> dict:
    """Execute one attempt end to end and return its raw-result/v1 record."""
    instance_id = task.instance_id
    slug = f"{instance_id}__pass-{pass_idx}"
    started_at = utcnow()
    started_mono = time.monotonic()
    deadline = started_mono + float(ctx.params["task_timeout_s"])

    traj_rel = f"trajectories/{instance_id}/pass-{pass_idx}.jsonl"
    patch_rel = f"patches/{instance_id}/pass-{pass_idx}.diff"
    traj_path = ctx.run_dir / traj_rel
    patch_path = ctx.run_dir / patch_rel
    log_path = ctx.run_dir / "logs" / "attempts" / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    traj = TrajectoryWriter(traj_path, int(ctx.cfg["loop"]["trajectory_content_max_bytes"]))
    log_handle = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        log_handle.write(f"{iso(utcnow())} {message}\n")
        log_handle.flush()

    executor: Executor | None = None
    terminal_code: str | None = None
    detail = ""
    patch = ""
    verdict: Any = None
    grader_error: str | None = None
    assistant_texts: list[str] = []

    try:
        log(f"attempt {slug} suite={ctx.suite} model={ctx.model}")
        prompt = ctx.adapter.build_prompt(task)
        if prompt.template_id != prompt_pkg.TEMPLATE_ID:
            # §5.2: all three adapters must render the same template. A mismatch means the
            # harness is no longer the control variable, so the attempt must not run.
            raise AttemptAbort(
                "INFRA_SANDBOX",
                f"adapter rendered template {prompt.template_id!r}, harness ships "
                f"{prompt_pkg.TEMPLATE_ID!r}",
            )
        log(f"prompt {prompt.template_id} sha256={prompt.prompt_sha256}")

        executor = build_workspace(task, slug, ctx.scratch_root)
        log(f"workspace ready: {type(executor).__name__} {executor.workdir}")
        toolbox = Toolbox(executor, task, ctx.cfg)

        terminal_code, detail = agent_loop(
            ctx, prompt, toolbox, stats, traj, log, deadline, assistant_texts
        )
        patch = capture_patch(executor)
        log(f"loop finished: terminal={terminal_code or 'self-terminated'} patch_bytes={len(patch)}")

        if terminal_code is None and not patch.strip() and looks_like_refusal(assistant_texts):
            terminal_code, detail = "MODEL_REFUSAL", "model declined the task instead of attempting it"

    except AttemptAbort as exc:
        terminal_code, detail = exc.code, exc.detail
        log(f"aborted: {exc.code}: {exc.detail}")
        traj.write(role="system", kind="error", content=f"{exc.code}: {exc.detail}")
        # A MODEL_* abort can still leave a usable patch on disk; §4 says the terminal code
        # stands "unless it resolves", which means it must still be graded.
        if exc.code.startswith("MODEL_") and executor is not None:
            try:
                patch = capture_patch(executor)
            except Exception:  # noqa: BLE001 — best effort; the abort code already stands
                patch = ""
    except SandboxError as exc:
        terminal_code, detail = "INFRA_SANDBOX", str(exc)[:512]
        log(f"sandbox failure: {exc}")
        traj.write(role="system", kind="error", content=f"INFRA_SANDBOX: {exc}")
    except Exception as exc:  # noqa: BLE001 — anything unmapped is INFRA_UNKNOWN, and a bug
        terminal_code, detail = "INFRA_UNKNOWN", f"{type(exc).__name__}: {exc}"[:512]
        log(f"unhandled: {type(exc).__name__}: {exc}")
        traj.write(role="system", kind="error", content=f"INFRA_UNKNOWN: {exc}")
    finally:
        # The workspace is not needed for grading: §5.3 requires grade() to build its own
        # environment, so the container/checkout is released as early as possible.
        if executor is not None:
            try:
                executor.cleanup()
            except Exception:  # noqa: BLE001
                pass

    # Grading. Skipped for SERVER_*/INFRA_* (the attempt never reached a gradable state) and
    # for MODEL_REFUSAL (no patch by definition); everything else is graded, so that a patch
    # produced under a budget or loop cut-off can still come back OK.
    gradable = terminal_code is None or terminal_code.startswith(("MODEL_", "BUDGET_"))
    if gradable and terminal_code != "MODEL_REFUSAL":
        verdict, grader_error = grade_patch(ctx, task, patch)
        if grader_error:
            log(f"grader failed: {grader_error}")
        else:
            log(f"graded: resolved={_verdict_field(verdict, 'resolved')} "
                f"code={_verdict_field(verdict, 'error_code')}")
    traj.close()
    log_handle.close()

    ended_at = utcnow()
    wall_clock_ms = int((time.monotonic() - started_mono) * 1000)

    # --- resolve the final error_code (CONTRACTS.md §4 precedence) ---------------------
    resolved = bool(_verdict_field(verdict, "resolved", False))
    if grader_error is not None:
        error_code = "INFRA_GRADER"
        detail = grader_error
        resolved = False
    elif resolved:
        # A budget/loop terminal condition does not cancel a patch that actually resolves.
        error_code = "OK"
        detail = (
            f"resolved despite {terminal_code}: {detail}"
            if terminal_code
            else str(_verdict_field(verdict, "detail", "") or "")
        )[:512]
    elif terminal_code is not None:
        error_code = terminal_code
    elif verdict is not None:
        error_code = str(_verdict_field(verdict, "error_code", "INFRA_UNKNOWN"))
        detail = str(_verdict_field(verdict, "detail", ""))[:512]
    else:
        error_code = "INFRA_UNKNOWN"
        detail = detail or "attempt produced neither a terminal code nor a verdict"
    if error_code not in ERROR_CODE_SET:
        detail = f"adapter returned unknown error_code {error_code!r}: {detail}"[:512]
        error_code = "INFRA_UNKNOWN"

    # --- patch artefact ---------------------------------------------------------------
    patch_block: dict[str, Any]
    if patch.strip():
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch, encoding="utf-8")
        files_changed, added, removed = patch_stats(patch)
        patch_block = {
            "present": True,
            "ref": patch_rel,
            "sha256": sha256_file(patch_path),
            "bytes": patch_path.stat().st_size,
            "files_changed": files_changed,
            "lines_added": added,
            "lines_removed": removed,
        }
    else:
        patch_block = {
            "present": False,
            "ref": None,
            "sha256": None,
            "bytes": None,
            "files_changed": None,
            "lines_added": None,
            "lines_removed": None,
        }

    trajectory_block = {
        "ref": traj_rel,
        "sha256": sha256_file(traj_path) if traj_path.exists() else None,
        "records": traj.records,
        "bytes": traj_path.stat().st_size if traj_path.exists() else 0,
        "consent_class": ctx.consent_class,
    }

    grade_block = None
    if verdict is not None and grader_error is None:
        grade_block = {
            "grader": str(_verdict_field(verdict, "grader", "unknown")),
            "grader_version": str(_verdict_field(verdict, "grader_version", "unknown")),
            "adapter_version": str(getattr(ctx.adapter, "ADAPTER_VERSION", "unknown")),
            "fail_to_pass": _verdict_field(verdict, "fail_to_pass", {"passed": 0, "total": 0}),
            "pass_to_pass": _verdict_field(verdict, "pass_to_pass", {"passed": 0, "total": 0}),
            "graded_at": iso(ended_at),
        }

    gpu_seconds = wall_clock_ms / 1000.0
    # Per-attempt cost is *attributable*, not billed (CONTRACTS.md §3.1/§8): the headline
    # number is computed at run level by analysis/aggregate.py from the manifest price. The
    # rate comes from the manifest that run.sh already resolved; 0 means it was unavailable,
    # and the record still carries gpu_seconds so cost can be recomputed later.
    cents_per_hour = ctx.cents_per_hour

    record = {
        "schema": "raw-result/v1",
        "run_id": ctx.run_id,
        "attempt_id": attempt_id(ctx.run_id, instance_id, pass_idx),
        "suite": ctx.suite,
        "instance_id": instance_id,
        "partition": ctx.partition_of(task),
        "model": ctx.model,
        "pass_idx": pass_idx,
        "started_at": iso(started_at),
        "ended_at": iso(ended_at),
        "wall_clock_ms": wall_clock_ms,
        "resolved": bool(resolved),
        "error_code": error_code,
        "error_detail": (detail or "")[:512],
        "tokens": {
            "prompt": stats.prompt_tokens,
            "completion": stats.completion_tokens,
            "total": stats.total_tokens,
            "cached_prompt": stats.cached_prompt_tokens,
        },
        "llm_calls": stats.llm_calls,
        "iterations": stats.iterations,
        "tool_calls": stats.tool_calls,
        "harness_retries": stats.retries,
        "latency_ms": {
            "generation_total": int(sum(stats.latencies_ms)),
            # Non-streaming client: time-to-first-token is not observable. Recorded as null
            # rather than guessed (CONTRACTS.md §3 permits absent optional detail).
            "ttft_p50": None,
            "ttft_max": None,
            "per_call_p50": _p50(stats.latencies_ms),
            "per_call_max": _pmax(stats.latencies_ms),
        },
        "patch": patch_block,
        "trajectory": trajectory_block,
        "grade": grade_block,
        "cost": {
            "gpu_seconds": round(gpu_seconds, 3),
            "effective_cents_per_hour": cents_per_hour,
            "usd": round(gpu_seconds / 3600.0 * cents_per_hour / 100.0, 6),
        },
    }
    if resolved and error_code != "OK":  # invariant from §3.1, checked rather than assumed
        record["error_code"] = "OK"
    return record


# --------------------------------------------------------------------------------------
# Suite / adapter / partition loading
# --------------------------------------------------------------------------------------


def load_adapter(suite: str) -> Any:
    """Resolve the suite adapter through the registry (CONTRACTS.md §5.4)."""
    if suite not in SUITE_MODULES:
        raise ConfigError(f"unknown suite {suite!r}; expected one of {', '.join(SUITES)}")
    registry_error: Exception | None = None
    try:
        registry = importlib.import_module("harness.adapters")
        getter = getattr(registry, "get", None)
        if callable(getter):
            return getter(suite)
        adapters = getattr(registry, "ADAPTERS", {})
        if suite in adapters:
            return adapters[suite]
    except Exception as exc:  # noqa: BLE001 — fall back to the direct import, then report
        registry_error = exc
    try:
        return importlib.import_module(f"harness.adapters.{SUITE_MODULES[suite]}")
    except Exception as exc:  # noqa: BLE001 — an unimportable adapter is a config error (exit 2)
        raise ConfigError(
            f"no usable adapter for suite {suite!r}: {type(exc).__name__}: {exc}"
            + (f" (registry: {type(registry_error).__name__}: {registry_error})" if registry_error else "")
        ) from exc


def adapter_path(adapter: Any) -> str:
    module_file = getattr(adapter, "__file__", "") or ""
    try:
        return str(Path(module_file).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return module_file


def default_seed_file(suite: str) -> Path:
    return REPO_ROOT / DEFAULT_SEED_FILES[suite]


def load_partitions(path: Path | None) -> dict[str, str]:
    """qualified_id -> partition name. A missing file is tolerated: everything is
    `unpartitioned`, which `run.sh` records and `aggregate.py` can see."""
    if path is None or not Path(path).exists():
        if path is not None:
            stderr(f"==> warning: partitions file {path} not found — tasks are unpartitioned")
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read partitions file {path}: {exc}") from exc
    mapping: dict[str, str] = {}
    for name, block in (data.get("partitions") or {}).items():
        for qualified in block.get("ids") or []:
            mapping[qualified] = name
    return mapping


def select_tasks(adapter: Any, seed_file: Path, limit: int | None, instances: list[str]) -> list[Any]:
    if not seed_file.exists():
        raise ConfigError(f"seed file not found: {seed_file}")
    try:
        tasks = list(adapter.load_tasks(seed_file))
    except Exception as exc:  # noqa: BLE001 — a bad seed file is a config error
        raise ConfigError(f"{adapter_path(adapter)}.load_tasks failed: {exc}") from exc
    if instances:
        wanted = set(instances)
        tasks = [t for t in tasks if t.instance_id in wanted]
        missing = wanted - {t.instance_id for t in tasks}
        if missing:
            raise ConfigError(f"instance id(s) not in the seed file: {sorted(missing)}")
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ConfigError("no tasks selected")
    return tasks


def existing_attempts(results_path: Path) -> set[tuple[str, int]]:
    """(instance_id, pass_idx) pairs already present, so --resume appends only what is missing."""
    done: set[tuple[str, int]] = set()
    if not results_path.exists():
        return done
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done.add((record["instance_id"], int(record["pass_idx"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


class ResultsWriter:
    """Append-only, flushed-and-fsynced after every record (CONTRACTS.md §0)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, record: dict) -> None:
        with self._lock:
            self._handle.write(jsonl_line(record))
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------
# `run` — orchestration
# --------------------------------------------------------------------------------------


def load_run_manifest(path: Path) -> dict:
    """Read the run manifest `run.sh` already wrote (CONTRACTS.md §2).

    The manifest — not the command line — is the authority for a real run: it is what the
    published results are described by, so the loop must execute exactly what it claims.
    """
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read run manifest {path}: {exc}") from exc
    if manifest.get("schema") != "run-manifest/v1":
        raise ConfigError(f"{path}: expected schema run-manifest/v1, got {manifest.get('schema')!r}")
    for required in ("run_id", "suite", "model", "inference"):
        if required not in manifest:
            raise ConfigError(f"{path}: manifest is missing {required!r}")
    return manifest


def apply_manifest(cfg: dict, manifest: dict) -> str:
    """Adopt the manifest's `inference` block as the config. Returns the endpoint."""
    block = dict(manifest["inference"])
    endpoint = block.pop("endpoint", DEFAULT_ENDPOINT)
    unknown = set(block) - set(DEFAULT_CONFIG["inference"])
    missing = set(DEFAULT_CONFIG["inference"]) - set(block)
    if unknown or missing:
        raise ConfigError(
            f"manifest inference block does not match CONTRACTS.md §2.1: "
            f"unknown={sorted(unknown)} missing={sorted(missing)}"
        )
    cfg["inference"] = block
    return endpoint


def read_id_list(path: Path) -> list[str]:
    """One instance id per line; blank lines and `#` comments ignored."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read instance list {path}: {exc}") from exc
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def order_by_manifest(tasks: list[Any], instance_ids: list[str]) -> list[Any]:
    """Restrict and order the loaded tasks to the manifest's instance list, verbatim."""
    by_id = {t.instance_id: t for t in tasks}
    missing = [i for i in instance_ids if i not in by_id]
    if missing:
        raise ConfigError(
            f"{len(missing)} manifest instance id(s) are not in the seed file, first few: "
            f"{missing[:5]}"
        )
    return [by_id[i] for i in instance_ids]


def check_tool_parity() -> None:
    """The tools the prompt advertises MUST be exactly the tools the harness implements.

    A model that is offered a tool the harness cannot run (or that cannot see one it could
    have used) is being run on a different harness, which is the one thing this study may
    not allow to vary.
    """
    advertised = {t.get("function", {}).get("name") for t in prompt_pkg.tools()}
    implemented = set(Toolbox.NAMES)
    if advertised != implemented:
        raise ConfigError(
            "tool set mismatch between harness/prompts/tools.json and agent.py: "
            f"advertised-only={sorted(advertised - implemented)}, "
            f"implemented-only={sorted(implemented - advertised)}"
        )


def cmd_run(args: argparse.Namespace) -> int:
    _install_signal_handlers()
    cfg = load_config(args.config)
    check_tool_parity()

    # `run.sh` drives this with --manifest: the manifest it already wrote is the authority
    # for model, suite, endpoint, instance list, consent class and every inference knob.
    manifest = load_run_manifest(Path(args.manifest)) if args.manifest else None
    endpoint = args.endpoint
    model = args.model
    suite = args.suite
    run_id = args.run_id
    manifest_instance_ids: list[str] = []
    cents_per_hour = float(os.environ.get("HARNESS_EFFECTIVE_CENTS_PER_HOUR", "0") or 0)
    consent_class = args.consent_class

    if manifest is not None:
        manifest_endpoint = apply_manifest(cfg, manifest)
        endpoint = args.endpoint if getattr(args, "endpoint_explicit", False) else manifest_endpoint
        model = model or manifest["model"]["name"]
        suite = suite or manifest["suite"]["name"]
        run_id = run_id or manifest["run_id"]
        manifest_instance_ids = list(manifest["suite"].get("instance_ids") or [])
        consent_class = consent_class or (manifest.get("flags") or {}).get("consent_class")
        cents_per_hour = float((manifest.get("price") or {}).get("effective_cents_per_hour") or cents_per_hour)
        if not args.seed_file and manifest["suite"].get("seed_file"):
            args.seed_file = str(REPO_ROOT / manifest["suite"]["seed_file"])
        if not args.partitions and manifest["suite"].get("partitions_file"):
            args.partitions = str(REPO_ROOT / manifest["suite"]["partitions_file"])
    if not model or not suite:
        raise ConfigError("--model and --suite are required unless --manifest supplies them")
    if suite not in SUITES:
        raise ConfigError(f"unknown suite {suite!r}; expected one of {', '.join(SUITES)}")

    params = inference_params(
        cfg,
        endpoint=endpoint,
        max_iters=args.max_iters,
        task_timeout_s=args.task_timeout,
        concurrency=args.concurrency,
        passes=args.passes,
    )
    for reason in nonconformant_reasons(cfg, params):
        stderr(f"==> NONCONFORMANT: {reason}")

    adapter = load_adapter(suite)
    seed_file = Path(args.seed_file) if args.seed_file else default_seed_file(suite)
    partitions_path = Path(args.partitions) if args.partitions else REPO_ROOT / "suites/partitions.json"
    tasks = select_tasks(adapter, seed_file, args.limit, args.instance or [])
    if manifest_instance_ids and not (args.instance or args.limit):
        tasks = order_by_manifest(tasks, manifest_instance_ids)
    if args.only_instances:
        wanted = set(read_id_list(Path(args.only_instances)))
        tasks = [t for t in tasks if t.instance_id in wanted]
        if not tasks:
            stderr("==> nothing to do: --only-instances matched no task in this suite")
    partitions = load_partitions(partitions_path)

    run_id = run_id or "{}__{}__{}__{}".format(
        model, suite, utcnow().strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:6]
    )
    run_dir = Path(args.run_dir) if args.run_dir else REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(args.scratch or os.environ.get("HARNESS_SCRATCH") or (run_dir / "workspaces"))
    scratch_root.mkdir(parents=True, exist_ok=True)

    consent_class = consent_class or getattr(adapter, "CONSENT_CLASS", "public")
    client = LLMClient(endpoint, model, cfg)

    if not args.no_preflight:
        try:
            served = client.health()
        except Exception as exc:  # noqa: BLE001
            stderr(f"==> preflight: endpoint {endpoint} unreachable: {exc}")
            return 3
        if model not in served:
            stderr(f"==> preflight: endpoint serves {served}, expected model {model!r}")
            return 3
        stderr(f"==> preflight ok: {endpoint} serving {served}")

    pass_indexes = [args.pass_idx] if args.pass_idx is not None else list(range(int(params["passes"])))
    results_path = run_dir / "results.jsonl"
    already = existing_attempts(results_path) if args.resume else set()
    work = [
        (task, pass_idx)
        for pass_idx in pass_indexes
        for task in tasks
        if (task.instance_id, pass_idx) not in already
    ]
    planned = len(work)
    stderr(
        f"==> run {run_id}: {len(tasks)} task(s) x {len(pass_indexes)} pass(es) = {planned} attempt(s)"
        f"{f', {len(already)} already done' if already else ''}, concurrency {params['concurrency']}"
    )

    ctx = RunContext(
        cfg=cfg,
        params=params,
        client=client,
        adapter=adapter,
        run_id=run_id,
        run_dir=run_dir,
        model=model,
        suite=suite,
        consent_class=consent_class,
        scratch_root=scratch_root,
        partitions=partitions,
        cents_per_hour=cents_per_hour,
    )

    writer = ResultsWriter(results_path)
    started_at = utcnow()
    counters: dict[str, int] = {}
    written = 0
    resolved_count = 0

    def execute(item: tuple[Any, int]) -> dict:
        task, pass_idx = item
        return run_attempt(ctx, task, pass_idx)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(params["concurrency"])) as pool:
            futures = {pool.submit(execute, item): item for item in work}
            for future in concurrent.futures.as_completed(futures):
                task, pass_idx = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 — never lose an attempt record
                    record = _fallback_record(ctx, task, pass_idx, exc)
                writer.append(record)
                written += 1
                counters[record["error_code"]] = counters.get(record["error_code"], 0) + 1
                resolved_count += 1 if record["resolved"] else 0
                stderr(
                    f"==> [{written}/{planned}] {record['instance_id']} pass-{pass_idx} "
                    f"{record['error_code']} resolved={record['resolved']} "
                    f"{record['wall_clock_ms'] // 1000}s iters={record['iterations']} "
                    f"tok={record['tokens']['total']}"
                )
    finally:
        writer.close()

    ended_at = utcnow()
    scored = sum(count for code, count in counters.items() if not code.startswith("INFRA_"))
    grader_failures = counters.get("INFRA_GRADER", 0)
    grading_degraded = written > 0 and grader_failures / written > 0.02

    if _STOP_SIGNAL:
        exit_code = 130 if _STOP_SIGNAL[0] == signal.SIGINT else 4
    elif written < planned:
        exit_code = 4
    elif grading_degraded:
        exit_code = 5
    else:
        exit_code = 0
    status = "complete" if exit_code == 0 else "incomplete"

    summary = {
        "schema": "agent-summary/v1",
        "run_id": run_id,
        "suite": suite,
        "model": model,
        "run_dir": str(run_dir),
        "status": status,
        "exit_code": exit_code,
        "attempts_planned": planned + len(already),
        "attempts_written": written + len(already),
        "attempts_scored": scored,
        "resolved": resolved_count,
        "error_codes": dict(sorted(counters.items())),
        "grading_degraded": grading_degraded,
        "infra_unknown": counters.get("INFRA_UNKNOWN", 0),
        "started_at": iso(started_at),
        "ended_at": iso(ended_at),
        "wall_clock_s": int((ended_at - started_at).total_seconds()),
        "nonconformant_reasons": nonconformant_reasons(cfg, params),
    }
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    stderr("==> summary " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return exit_code


def _fallback_record(ctx: RunContext, task: Any, pass_idx: int, exc: BaseException) -> dict:
    """A record must exist for every planned attempt, even when run_attempt itself died."""
    now = iso(utcnow())
    return {
        "schema": "raw-result/v1",
        "run_id": ctx.run_id,
        "attempt_id": attempt_id(ctx.run_id, task.instance_id, pass_idx),
        "suite": ctx.suite,
        "instance_id": task.instance_id,
        "partition": ctx.partition_of(task),
        "model": ctx.model,
        "pass_idx": pass_idx,
        "started_at": now,
        "ended_at": now,
        "wall_clock_ms": 0,
        "resolved": False,
        "error_code": "INFRA_UNKNOWN",
        "error_detail": f"attempt worker raised {type(exc).__name__}: {exc}"[:512],
        "tokens": {"prompt": 0, "completion": 0, "total": 0, "cached_prompt": 0},
        "llm_calls": 0,
        "iterations": 0,
        "tool_calls": 0,
        "harness_retries": 0,
        "latency_ms": {
            "generation_total": 0,
            "ttft_p50": None,
            "ttft_max": None,
            "per_call_p50": None,
            "per_call_max": None,
        },
        "patch": {
            "present": False,
            "ref": None,
            "sha256": None,
            "bytes": None,
            "files_changed": None,
            "lines_added": None,
            "lines_removed": None,
        },
        "trajectory": {
            "ref": f"trajectories/{task.instance_id}/pass-{pass_idx}.jsonl",
            "sha256": None,
            "records": 0,
            "bytes": 0,
            "consent_class": ctx.consent_class,
        },
        "grade": None,
        "cost": {"gpu_seconds": 0.0, "effective_cents_per_hour": 0.0, "usd": 0.0},
    }


# --------------------------------------------------------------------------------------
# Introspection subcommands — what run.sh needs to build the manifest
# --------------------------------------------------------------------------------------


def emit(obj: Any) -> None:
    """The only thing these subcommands ever put on stdout: one canonical JSON object."""
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_inference_params(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    params = inference_params(
        cfg,
        endpoint=args.endpoint,
        max_iters=args.max_iters,
        task_timeout_s=args.task_timeout,
        concurrency=args.concurrency,
        passes=args.passes,
    )
    if args.with_provenance:
        emit(
            {
                "inference": params,
                "agent_config": str(args.config or CONFIG_PATH),
                "agent_config_sha256": sha256_file(Path(args.config or CONFIG_PATH))
                if Path(args.config or CONFIG_PATH).exists()
                else None,
                "prompt_template_id": prompt_pkg.TEMPLATE_ID,
                "prompt_dir_sha256": prompt_pkg.dir_sha256(),
                "nonconformant_reasons": nonconformant_reasons(cfg, params),
            }
        )
    else:
        emit(params)
    return 0


def cmd_prompt_info(args: argparse.Namespace) -> int:
    check_tool_parity()
    info = prompt_pkg.info()
    info["agent_config_sha256"] = (
        sha256_file(Path(args.config or CONFIG_PATH)) if Path(args.config or CONFIG_PATH).exists() else None
    )
    emit(info)
    return 0


def cmd_adapter_info(args: argparse.Namespace) -> int:
    adapter = load_adapter(args.suite)
    path = adapter_path(adapter)
    absolute = REPO_ROOT / path
    emit(
        {
            "suite": args.suite,
            "adapter": path,
            "adapter_version": getattr(adapter, "ADAPTER_VERSION", None),
            "adapter_sha256": sha256_file(absolute) if absolute.exists() else None,
            "consent_class": getattr(adapter, "CONSENT_CLASS", "public"),
            "suite_name": getattr(adapter, "SUITE_NAME", None),
            "default_seed_file": DEFAULT_SEED_FILES[args.suite],
            "environment_digest": _safe_call(adapter, "environment_digest"),
        }
    )
    return 0


def _safe_call(module: Any, name: str) -> Any:
    func = getattr(module, name, None)
    if not callable(func):
        return None
    try:
        return func()
    except Exception as exc:  # noqa: BLE001 — introspection must never crash run.sh
        return f"unavailable: {type(exc).__name__}: {exc}"


def cmd_prompt_preview(args: argparse.Namespace) -> int:
    """Render the first selected task's prompt — `run.sh --dry-run` writes this to disk."""
    adapter = load_adapter(args.suite)
    seed_file = Path(args.seed_file) if args.seed_file else default_seed_file(args.suite)
    tasks = select_tasks(adapter, seed_file, 1, args.instance or [])
    task = tasks[0]
    prompt = adapter.build_prompt(task)
    if prompt.template_id != prompt_pkg.TEMPLATE_ID:
        stderr(
            f"==> template id mismatch: adapter rendered {prompt.template_id!r}, harness ships "
            f"{prompt_pkg.TEMPLATE_ID!r}"
        )
        return 2
    text = "\n".join(
        [
            f"# template_id : {prompt.template_id}",
            f"# prompt_sha256: {prompt.prompt_sha256}",
            f"# suite        : {args.suite}",
            f"# instance_id  : {task.instance_id}",
            f"# tools        : {', '.join(t.get('function', {}).get('name', '?') for t in prompt.tools)}",
            "",
            "===== system =====",
            prompt.system,
            "",
            "===== user =====",
            prompt.user,
            "",
        ]
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        stderr(f"==> wrote {args.out}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness/agent.py",
        description="The fixed agent loop (AgentTask AI-P153). See docs/CONTRACTS.md.",
    )
    parser.add_argument("--config", default=None, help=f"agent config (default: {CONFIG_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_knob_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--endpoint",
            default=os.environ.get("HARNESS_ENDPOINT", DEFAULT_ENDPOINT),
            help="OpenAI-compatible base URL; the manifest's endpoint wins unless this is passed",
        )
        p.add_argument("--max-iters", type=int, default=None, help="HELD CONSTANT — overriding is nonconformant")
        p.add_argument("--task-timeout", type=int, default=None, help="per-attempt wall-clock ceiling, seconds")
        p.add_argument("--concurrency", type=int, default=None)
        p.add_argument("--passes", type=int, default=None)

    run_p = sub.add_parser("run", help="execute attempts and append raw-result/v1 records")
    run_p.add_argument(
        "--manifest",
        default=None,
        help="run-manifest.json written by run.sh — the authority for every knob below",
    )
    run_p.add_argument("--model", default=None, help="required unless --manifest supplies it")
    run_p.add_argument("--suite", default=None, choices=SUITES, help="required unless --manifest supplies it")
    run_p.add_argument("--run-id", default=None, help="run id from run.sh; generated if omitted")
    run_p.add_argument("--run-dir", default=None, help="run directory (must already contain the manifest)")
    run_p.add_argument("--pass-idx", type=int, default=None, help="run exactly this pass instead of all")
    run_p.add_argument("--seed-file", default=None)
    run_p.add_argument("--partitions", default=None)
    run_p.add_argument("--limit", type=int, default=None, help="debug: first N instances in seed order")
    run_p.add_argument("--instance", action="append", default=[], help="debug: repeatable instance id")
    run_p.add_argument("--resume", action="store_true", help="skip (instance, pass) pairs already recorded")
    run_p.add_argument(
        "--only-instances",
        default=None,
        help="file of instance ids (one per line) to run — how run.sh resumes a pass",
    )
    run_p.add_argument("--consent-class", default=None, choices=["public", "restricted"])
    run_p.add_argument("--scratch", default=None, help="root for per-attempt workspaces")
    run_p.add_argument("--summary-out", default=None, help="write the JSON run summary here")
    run_p.add_argument("--no-preflight", action="store_true", help="skip the GET /models check")
    add_knob_flags(run_p)
    run_p.set_defaults(func=cmd_run)

    params_p = sub.add_parser("inference-params", help="emit the manifest `inference` block")
    params_p.add_argument("--with-provenance", action="store_true", help="also emit prompt/config hashes")
    add_knob_flags(params_p)
    params_p.set_defaults(func=cmd_inference_params)

    info_p = sub.add_parser("prompt-info", help="template id, prompt_dir_sha256, tools, variables")
    info_p.set_defaults(func=cmd_prompt_info)

    adapter_p = sub.add_parser("adapter-info", help="adapter path/version/sha256 for the manifest")
    adapter_p.add_argument("--suite", required=True, choices=SUITES)
    adapter_p.set_defaults(func=cmd_adapter_info)

    preview_p = sub.add_parser("prompt-preview", help="render the first task's prompt (--dry-run)")
    preview_p.add_argument("--suite", required=True, choices=SUITES)
    preview_p.add_argument("--seed-file", default=None)
    preview_p.add_argument("--instance", action="append", default=[])
    preview_p.add_argument("--out", default=None, help="write here instead of stdout")
    preview_p.set_defaults(func=cmd_prompt_preview)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    # The manifest's endpoint is authoritative unless the caller explicitly overrode it.
    args.endpoint_explicit = any(a == "--endpoint" or a.startswith("--endpoint=") for a in raw)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        stderr(f"error: {exc}")
        return 2
    except prompt_pkg.PromptError as exc:
        stderr(f"error: prompt/template: {exc}")
        return 2
    except KeyboardInterrupt:
        stderr("==> interrupted")
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
