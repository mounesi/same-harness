#!/usr/bin/env python3
# aggregate.py — publication tables for "The Harness Variable" (AgentTask AI-P153).
#
#   python3 analysis/aggregate.py --manifests RUN_A/run-manifest.json RUN_B/run-manifest.json
#   python3 analysis/aggregate.py --manifest-list runs.txt --results-root ~/unpacked
#   python3 analysis/aggregate.py --manifest-list runs.txt --api-pricing analysis/api-pricing.json \
#                                 --out-dir analysis/tables --print md
#
# Consumes an EXPLICIT list of run manifests (never an ambient results/ directory — a
# directory argument is refused by design, CONTRACTS.md §6.2 rule 4 applied to analysis),
# locates each run's results.jsonl, validates schema versions and SHA256SUMS, then emits:
#
#   summary.md            paste-ready markdown tables
#   summary.json          machine-readable everything (schema aggregate-report/v1)
#   by_model_suite.csv    one row per (model, suite): the headline table
#   failures.csv          §4 failure-taxonomy breakdown, long form
#   contamination.csv     per-model Verified vs Pro vs AgentTask deltas
#   runs.csv              run inventory / provenance
#
# Headline metric is cost per resolved task (CONTRACTS.md §8). GPU-served models are priced
# from the manifest price snapshot × instance-hours; API-baseline models are priced per token
# from --api-pricing. Runs that differ in ANY verdict-affecting harness knob (version, prompt
# hash/template, agent config, adapter version, grading environment digest, sampling params,
# iteration/token budgets, max_model_len, serving EXTRA_ARGS, task timeout) are REFUSED unless --allow-mixed is
# passed, and then every output is loudly annotated.
#
# Two independent manifest flags gate inclusion (CONTRACTS.md §2.2):
#   flags.nonconformant         — a genuine harness deviation that breaks comparability
#                                 (non-default budget, dirty repo, unresolved weight revision,
#                                 prompt/template drift). EXCLUDED by default;
#                                 --include-nonconformant overrides.
#   flags.provenance_incomplete — cost/provenance attribution is imprecise but the science is
#                                 intact (missing instance id/region, fallback pricing,
#                                 unresolved lock hash). INCLUDED by default; the cost columns
#                                 for the affected groups are annotated approximate (≈) and the
#                                 unresolved fields are listed per run.
# A pre-split manifest that carries only the old single `nonconformant` flag is treated as
# nonconformant (the conservative reading) and is called out as legacy wherever it appears.
#
# Exit codes: 0 ok · 1 usage · 2 validation/comparability refusal · 3 missing or unreadable input
#
# stdlib only. Python 3.11.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

AGGREGATOR_VERSION = "1.0.0"
REPORT_SCHEMA = "aggregate-report/v1"

KNOWN_MANIFEST_SCHEMAS = {"run-manifest/v1"}
KNOWN_RESULT_SCHEMAS = {"raw-result/v1"}
KNOWN_API_PRICING_SCHEMAS = {"api-pricing/v1"}

# CONTRACTS.md §4 — closed enum, 18 values.
ERROR_CODES = (
    "OK",
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
    "SERVER_ERROR",
    "SERVER_UNAVAILABLE",
    "INFRA_SANDBOX",
    "INFRA_GRADER",
    "INFRA_HOST",
    "INFRA_UNKNOWN",
)

# Grouped families used for the compact markdown failure table.
FAMILIES = (
    ("resolved", ("OK",)),
    ("no_patch", ("NO_PATCH",)),
    ("patch_malformed", ("PATCH_MALFORMED",)),
    ("tests_fail", ("TESTS_FAIL",)),
    ("tests_regression", ("TESTS_REGRESSION",)),
    ("budget", ("BUDGET_ITERATIONS", "BUDGET_TOKENS", "BUDGET_WALLCLOCK")),
    (
        "model",
        (
            "MODEL_CONTEXT_OVERFLOW",
            "MODEL_MALFORMED_TOOL_CALL",
            "MODEL_LOOP",
            "MODEL_REFUSAL",
            "MODEL_EMPTY_RESPONSE",
        ),
    ),
    ("server", ("SERVER_ERROR", "SERVER_UNAVAILABLE")),
    ("infra*", ("INFRA_SANDBOX", "INFRA_GRADER", "INFRA_HOST", "INFRA_UNKNOWN")),
)

SUITE_ORDER = ("swebench-verified", "swebench-pro", "agenttask")
SUITE_SHORT = {"swebench-verified": "Verified", "swebench-pro": "Pro", "agenttask": "AgentTask"}

# Fixed so a re-run of the aggregator reproduces the same confidence intervals byte for byte.
BOOTSTRAP_SEED = 20260830

# Thresholds from CONTRACTS.md §4 / §8.
INFRA_UNKNOWN_INVALID_SHARE = 0.02
GRADER_DEGRADED_SHARE = 0.02
SERVER_UNAVAILABLE_FLAG_SHARE = 0.05


# ---------------------------------------------------------------- small utilities


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dig(obj, *path, default=None):
    """Nested dict lookup that tolerates missing keys and explicit nulls."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur


def percentile(values: list[float], q: float):
    """Linear-interpolation percentile on an already-sorted list. q in [0,1]."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    k = (len(values) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(values[int(k)])
    return float(values[lo]) * (hi - k) + float(values[hi]) * (k - lo)


def _mean(values) -> float:
    """Arithmetic mean. Spelled out rather than statistics.fmean so this file also runs on
    the older interpreters that ship on some dev machines."""
    return sum(float(v) for v in values) / len(values)


def median_or_none(values: list[float]):
    return float(statistics.median(values)) if values else None


def mean_or_none(values: list[float]):
    return _mean(values) if values else None


def bootstrap_ci_over_passes(pass_rates, iters, alpha: float = 0.05):
    """RETIRED 2026-08-30 — kept for reference only; do not call.

    Bootstrapping across passes assumes the passes are independent draws. They are not:
    the study decodes greedily at temperature 0.0 with one held-constant seed, so the
    passes differ only by serving nondeterminism. Report the observed range instead.
    Re-enable this ONLY together with temperature > 0 and a per-pass seed.
    """
    """Percentile bootstrap of the mean pass-level resolve rate (CONTRACTS.md §8).

    With only 3 passes this interval is coarse by construction; the pass min/max range is
    reported alongside it and is the honest thing to quote in prose.
    """
    if len(pass_rates) < 2 or iters <= 0:
        return (None, None)
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(pass_rates)
    means = [_mean(rng.choices(pass_rates, k=n)) for _ in range(iters)]
    means.sort()
    return (percentile(means, alpha / 2), percentile(means, 1 - alpha / 2))


def bootstrap_ci_over_instances(per_instance: list[tuple[int, int]], iters: int, alpha: float = 0.05):
    """Cluster bootstrap: resample instances (not attempts), recompute the pooled rate.

    per_instance is a list of (resolved_count, scored_count) per instance id.
    """
    clusters = [c for c in per_instance if c[1] > 0]
    if len(clusters) < 2 or iters <= 0:
        return (None, None)
    rng = random.Random(BOOTSTRAP_SEED + 1)
    n = len(clusters)
    rates = []
    for _ in range(iters):
        sample = rng.choices(clusters, k=n)
        res = sum(c[0] for c in sample)
        sco = sum(c[1] for c in sample)
        if sco:
            rates.append(res / sco)
    if len(rates) < 2:
        return (None, None)
    rates.sort()
    return (percentile(rates, alpha / 2), percentile(rates, 1 - alpha / 2))


# ---------------------------------------------------------------- diagnostics


class Diagnostics:
    """Collects warnings/errors so every output can carry the same annotation set."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, level: str, code: str, message: str, **ctx) -> None:
        entry = {"level": level, "code": code, "message": message}
        if ctx:
            entry["context"] = ctx
        self.items.append(entry)

    def warn(self, code: str, message: str, **ctx) -> None:
        self.add("warning", code, message, **ctx)

    def error(self, code: str, message: str, **ctx) -> None:
        self.add("error", code, message, **ctx)

    @property
    def errors(self) -> list[dict]:
        return [i for i in self.items if i["level"] == "error"]

    @property
    def warnings(self) -> list[dict]:
        return [i for i in self.items if i["level"] == "warning"]


# ---------------------------------------------------------------- input loading


def read_manifest_paths(args, diag: Diagnostics) -> list[Path]:
    paths: list[Path] = []
    for raw in args.manifests or []:
        paths.append(Path(raw))
    for list_file in args.manifest_list or []:
        lf = Path(list_file)
        if not lf.is_file():
            diag.error("manifest_list_missing", f"--manifest-list not found: {lf}")
            continue
        for line in lf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            paths.append(p if p.is_absolute() else (lf.parent / p))
    return paths


def load_manifest(path: Path, diag: Diagnostics):
    if path.is_dir():
        diag.error(
            "directory_argument_refused",
            f"{path} is a directory. aggregate.py consumes an explicit list of run "
            "manifests; globbing an ambient results/ directory is refused by design.",
        )
        return None
    if not path.is_file():
        diag.error("manifest_missing", f"manifest not found: {path}")
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        diag.error("manifest_unreadable", f"{path}: {exc}")
        return None
    if not isinstance(manifest, dict):
        diag.error("manifest_malformed", f"{path}: top-level value is not an object")
        return None
    schema = manifest.get("schema")
    if schema not in KNOWN_MANIFEST_SCHEMAS:
        diag.error(
            "manifest_schema_unknown",
            f"{path}: schema {schema!r} not in {sorted(KNOWN_MANIFEST_SCHEMAS)}",
        )
        return None
    if not manifest.get("run_id"):
        diag.error("manifest_no_run_id", f"{path}: missing run_id")
        return None
    return manifest


def locate_run_dir(manifest_path: Path, run_id: str, roots: list[Path]) -> Path | None:
    """Find the directory holding results.jsonl for this run.

    1. next to the manifest (the run dir itself, or an unpacked bundle)
    2. <root>/<run_id>/ and <root>/runs/<run_id>/ for each --results-root
    """
    candidates = [manifest_path.parent, manifest_path.parent / run_id]
    for root in roots:
        candidates.append(root / run_id)
        candidates.append(root / "runs" / run_id)
        candidates.append(root)
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if (cand / "results.jsonl").is_file():
            return cand
    return None


def parse_sha256sums(path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
        digest, rel = parts[0].strip(), parts[1].strip().lstrip("*")
        table[rel] = digest.lower()
    return table


def verify_run_checksums(run_dir: Path, args, diag: Diagnostics, run_id: str) -> str:
    """Returns one of: verified | absent | skipped | failed."""
    if args.no_verify_checksums:
        return "skipped"
    sums_path = run_dir / "SHA256SUMS"
    if not sums_path.is_file():
        msg = f"{run_id}: SHA256SUMS absent in {run_dir} — bundle integrity unverified"
        if args.require_checksums:
            diag.error("checksums_absent", msg)
            return "failed"
        diag.warn("checksums_absent", msg)
        return "absent"
    table = parse_sha256sums(sums_path)
    status = "verified"
    want = table.get("results.jsonl")
    if want is None:
        diag.warn("checksums_no_results_entry", f"{run_id}: SHA256SUMS has no results.jsonl entry")
        status = "absent"
    else:
        got = sha256_file(run_dir / "results.jsonl")
        if got != want:
            diag.error(
                "checksum_mismatch",
                f"{run_id}: results.jsonl sha256 {got} != SHA256SUMS {want}",
            )
            return "failed"
    # The manifest is listed in SHA256SUMS too, but it is legitimately rewritten at end of
    # run, so a mismatch there is a warning rather than a hard failure.
    mwant = table.get("run-manifest.json")
    if mwant and (run_dir / "run-manifest.json").is_file():
        mgot = sha256_file(run_dir / "run-manifest.json")
        if mgot != mwant:
            diag.warn(
                "manifest_checksum_mismatch",
                f"{run_id}: run-manifest.json sha256 differs from SHA256SUMS "
                "(expected if the manifest was rewritten after the sums were taken)",
            )
    if args.verify_refs:
        missing = 0
        bad = 0
        for rel, want_hex in table.items():
            if not (rel.startswith("patches/") or rel.startswith("trajectories/")):
                continue
            fp = run_dir / rel
            if not fp.is_file():
                missing += 1
                continue
            if sha256_file(fp) != want_hex:
                bad += 1
        if missing or bad:
            diag.error(
                "refs_verification_failed",
                f"{run_id}: {missing} referenced artifact(s) missing, {bad} with wrong sha256",
            )
            return "failed"
    return status


def load_records(run_dir: Path, manifest: dict, args, diag: Diagnostics):
    """Read and validate results.jsonl. Returns (records, ok)."""
    run_id = manifest["run_id"]
    path = run_dir / "results.jsonl"
    records: list[dict] = []
    seen_keys: set[tuple[str, int]] = set()
    fatal = False

    def problem(code: str, message: str) -> None:
        nonlocal fatal
        if args.lenient:
            diag.warn(code, message)
        else:
            diag.error(code, message)
            fatal = True

    suite_ids = set(dig(manifest, "suite", "instance_ids", default=[]) or [])
    planned_passes = dig(manifest, "inference", "passes", default=None)
    unknown_id_reported = 0

    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        diag.error("results_unreadable", f"{run_id}: {exc}")
        return [], False

    for lineno, line in enumerate(raw_lines, 1):
        if not line.strip():
            problem("results_blank_line", f"{run_id}:{lineno}: blank line in results.jsonl")
            continue
        try:
            rec = json.loads(line)
        except ValueError as exc:
            problem("record_unparseable", f"{run_id}:{lineno}: {exc}")
            continue
        if not isinstance(rec, dict):
            problem("record_malformed", f"{run_id}:{lineno}: record is not an object")
            continue
        schema = rec.get("schema")
        if schema not in KNOWN_RESULT_SCHEMAS:
            problem(
                "result_schema_unknown",
                f"{run_id}:{lineno}: schema {schema!r} not in {sorted(KNOWN_RESULT_SCHEMAS)}",
            )
            continue
        if rec.get("run_id") != run_id:
            problem(
                "record_run_id_mismatch",
                f"{run_id}:{lineno}: record run_id {rec.get('run_id')!r} != manifest run_id",
            )
            continue
        code = rec.get("error_code")
        if code not in ERROR_CODES:
            problem(
                "error_code_unknown",
                f"{run_id}:{lineno}: error_code {code!r} is not one of the 18 values in §4",
            )
            continue
        if rec.get("resolved") and code != "OK":
            problem(
                "resolved_error_code_conflict",
                f"{run_id}:{lineno}: resolved=true with error_code={code!r} (§3.1 requires OK)",
            )
        if code == "OK" and not rec.get("resolved"):
            problem(
                "ok_without_resolved",
                f"{run_id}:{lineno}: error_code=OK with resolved=false is impossible (§4)",
            )
        instance_id = rec.get("instance_id")
        pass_idx = rec.get("pass_idx")
        if not isinstance(instance_id, str) or not isinstance(pass_idx, int):
            problem("record_key_invalid", f"{run_id}:{lineno}: bad instance_id/pass_idx")
            continue
        key = (instance_id, pass_idx)
        if key in seen_keys:
            problem(
                "duplicate_attempt",
                f"{run_id}: duplicate attempt for ({instance_id}, pass {pass_idx})",
            )
            continue
        seen_keys.add(key)
        if suite_ids and instance_id not in suite_ids:
            unknown_id_reported += 1
        if isinstance(planned_passes, int) and not (0 <= pass_idx < planned_passes):
            diag.warn(
                "pass_idx_out_of_range",
                f"{run_id}: pass_idx {pass_idx} outside 0..{planned_passes - 1}",
            )
        records.append(rec)

    if unknown_id_reported:
        diag.warn(
            "instance_not_in_seed",
            f"{run_id}: {unknown_id_reported} record(s) reference instance ids not listed in "
            "the manifest's suite.instance_ids",
        )

    written = dig(manifest, "timing", "attempts_written", default=None)
    if isinstance(written, int) and written != len(records):
        diag.warn(
            "attempts_written_mismatch",
            f"{run_id}: manifest timing.attempts_written={written} but results.jsonl has "
            f"{len(records)} valid record(s)",
        )
    if not records:
        diag.warn("run_empty", f"{run_id}: no usable records")
    return records, not fatal


def load_api_pricing(path: Path | None, diag: Diagnostics) -> dict:
    if path is None:
        return {}
    if not path.is_file():
        diag.error("api_pricing_missing", f"--api-pricing file not found: {path}")
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        diag.error("api_pricing_unreadable", f"{path}: {exc}")
        return {}
    if doc.get("schema") not in KNOWN_API_PRICING_SCHEMAS:
        diag.error(
            "api_pricing_schema_unknown",
            f"{path}: schema {doc.get('schema')!r} not in {sorted(KNOWN_API_PRICING_SCHEMAS)}",
        )
        return {}
    models = doc.get("models")
    if not isinstance(models, dict):
        diag.error("api_pricing_malformed", f"{path}: 'models' must be an object")
        return {}
    return doc


def load_cost_logs(paths: list[Path], diag: Diagnostics) -> dict[str, float]:
    """CI's cost-log.jsonl → setup USD per run_id. Reported separately, never folded in (§8)."""
    setup: dict[str, float] = {}
    for path in paths:
        if not path.is_file():
            diag.warn("cost_log_missing", f"--cost-log not found: {path}")
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                diag.warn("cost_log_unparseable", f"{path}:{lineno}: not JSON, skipped")
                continue
            if not isinstance(rec, dict):
                continue
            run_id = rec.get("run_id")
            if not isinstance(run_id, str):
                continue
            usd = rec.get("setup_cost_usd", rec.get("setup_usd"))
            if usd is None:
                seconds = rec.get("setup_seconds", rec.get("setup_s", rec.get("download_s")))
                cph = rec.get("effective_cents_per_hour", rec.get("price_cents_per_hour"))
                if isinstance(seconds, (int, float)) and isinstance(cph, (int, float)):
                    usd = seconds / 3600.0 * cph / 100.0
            if isinstance(usd, (int, float)):
                setup[run_id] = setup.get(run_id, 0.0) + float(usd)
    return setup


# ---------------------------------------------------------------- manifest flags
#
# CONTRACTS.md §2.2. The two flags mean different things and are handled differently:
# `nonconformant` excludes, `provenance_incomplete` annotates. Both are write-once booleans
# with an accompanying list of reasons. Manifests written before the split carry only
# `nonconformant`; those are handled by _LEGACY below.

NONCONFORMANT_REASON_KEYS = (
    "nonconformant_reasons",
    "nonconformant_reason",
)
PROVENANCE_REASON_KEYS = (
    "provenance_incomplete_reasons",
    "provenance_incomplete_reason",
    "provenance_unresolved",
    "provenance_unresolved_fields",
)

_LEGACY_NOTE = (
    "legacy pre-split manifest: it carries only the old single `nonconformant` flag, which "
    "conflated harness deviations with provenance gaps — this run is treated as a harness "
    "deviation (the conservative reading). Re-emit the manifest to separate the two."
)


def _reason_list(flags: dict, keys) -> list:
    """Collect reason strings from whichever of `keys` the manifest happens to use."""
    out: list = []
    for key in keys:
        val = flags.get(key)
        if isinstance(val, str):
            val = [val]
        if isinstance(val, (list, tuple)):
            for item in val:
                text = str(item).strip()
                if text and text not in out:
                    out.append(text)
    return out


def flag_state(manifest: dict) -> dict:
    """Normalise a manifest's conformance flags into the post-split shape.

    Returns keys: nonconformant, nonconformant_reasons, provenance_incomplete,
    provenance_incomplete_reasons, legacy_single_flag.
    """
    flags = manifest.get("flags")
    if not isinstance(flags, dict):
        flags = {}
    legacy = "provenance_incomplete" not in flags
    nonconformant = bool(flags.get("nonconformant", False))
    provenance = bool(flags.get("provenance_incomplete", False))
    n_reasons = _reason_list(flags, NONCONFORMANT_REASON_KEYS)
    p_reasons = _reason_list(flags, PROVENANCE_REASON_KEYS)
    if legacy and nonconformant and not n_reasons:
        # Pre-split run.sh recorded the reasons in `notes` (CONTRACTS.md §2.3).
        notes = manifest.get("notes")
        if isinstance(notes, str) and notes.strip():
            n_reasons = [part.strip() for part in notes.split(";") if part.strip()]
    if legacy and nonconformant:
        n_reasons = n_reasons + [_LEGACY_NOTE]
    return {
        "nonconformant": nonconformant,
        "nonconformant_reasons": n_reasons,
        "provenance_incomplete": provenance,
        "provenance_incomplete_reasons": p_reasons,
        "legacy_single_flag": legacy,
    }


def _reasons_suffix(reasons: list) -> str:
    return (" (" + "; ".join(reasons) + ")") if reasons else ""


# ---------------------------------------------------------------- run selection


def run_price_cph(manifest: dict, diag: Diagnostics):
    """effective cents/hour for a GPU-served run, with the documented fallback."""
    cph = dig(manifest, "price", "effective_cents_per_hour", default=None)
    if isinstance(cph, (int, float)):
        return float(cph)
    base = dig(manifest, "price", "price_cents_per_hour", default=None)
    nodes = dig(manifest, "price", "node_count", default=dig(manifest, "hardware", "node_count", default=1))
    if isinstance(base, (int, float)) and isinstance(nodes, (int, float)):
        diag.warn(
            "price_effective_missing",
            f"{manifest['run_id']}: price.effective_cents_per_hour missing; derived "
            f"{base} × {nodes} nodes",
        )
        return float(base) * float(nodes)
    return None


_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def _endpoint_is_loopback(endpoint) -> bool:
    """True when inference.endpoint points at this host — the self-hosted vLLM case."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        return False
    try:
        from urllib.parse import urlsplit

        host = urlsplit(endpoint.strip() if "://" in endpoint else "http://" + endpoint.strip()).hostname
    except ValueError:
        return False
    host = (host or "").lower()
    if host in _LOOPBACK_HOSTS:
        return True
    try:  # whole 127.0.0.0/8 and ::1 — the same rule harness/manifest.py applies
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def billing_mode_for(manifest: dict, api_pricing: dict) -> str:
    """Decide how a run is billed: "instance_hours" (GPU rented by the hour) or "per_token".

    Precedence (README "Cost"):
      1. price.billing_mode as written by harness/manifest.py — authoritative.
      2. Legacy manifests (no billing_mode): evidence of a rented instance wins —
         a price per hour, a lambda_instance_id, or a loopback inference endpoint —
         because the study serves some models BOTH ways and the model NAME alone cannot
         tell a self-hosted Qwen run from the API baseline of the same model.
      3. Inline per-token rates in the manifest → per_token.
      4. Last resort: the model name appears in --api-pricing → per_token.
      5. Otherwise instance_hours (the study's default); a missing price then surfaces
         as a cost diagnostic instead of being silently priced at API token rates.
    """
    declared = dig(manifest, "price", "billing_mode", default=None)
    if declared in ("per_token", "instance_hours"):
        return declared
    for key in ("effective_cents_per_hour", "price_cents_per_hour"):
        if isinstance(dig(manifest, "price", key, default=None), (int, float)):
            return "instance_hours"
    if dig(manifest, "hardware", "lambda_instance_id", default=None):
        return "instance_hours"
    if _endpoint_is_loopback(dig(manifest, "inference", "endpoint", default=None)):
        return "instance_hours"
    if isinstance(dig(manifest, "price", "input_usd_per_mtok", default=None), (int, float)):
        return "per_token"
    models = api_pricing.get("models", {}) if api_pricing else {}
    for key in (
        dig(manifest, "model", "name", default=None),
        dig(manifest, "model", "served_model_name", default=None),
        dig(manifest, "model", "hf_repo", default=None),
    ):
        if key and key in models:
            return "per_token"
    return "instance_hours"


def api_rates_for(manifest: dict, api_pricing: dict):
    """Return (rates_dict, source) or (None, None)."""
    price = manifest.get("price") or {}
    if price.get("billing_mode") == "per_token" and "input_usd_per_mtok" in price:
        return (
            {
                "input_usd_per_mtok": price.get("input_usd_per_mtok"),
                "output_usd_per_mtok": price.get("output_usd_per_mtok"),
                "cached_input_usd_per_mtok": price.get("cached_input_usd_per_mtok"),
                "provider": price.get("provider"),
            },
            "manifest",
        )
    models = api_pricing.get("models", {}) if api_pricing else {}
    for key in (
        dig(manifest, "model", "name", default=None),
        dig(manifest, "model", "served_model_name", default=None),
        dig(manifest, "model", "hf_repo", default=None),
    ):
        if key and key in models:
            return models[key], "api-pricing-file"
    return (None, None)


def select_runs(loaded: list[dict], args, diag: Diagnostics):
    """Split loaded runs into included / excluded per CONTRACTS.md §8 eligibility rules.

    `flags.nonconformant` excludes (it means the harness itself deviated, so the run is not
    comparable). `flags.provenance_incomplete` does NOT exclude — the science is intact and
    only the cost/provenance attribution is imprecise — but it is recorded so every cost
    column derived from that run can be annotated approximate.
    """
    included, excluded = [], []
    for run in loaded:
        m = run["manifest"]
        fs = flag_state(m)
        run["flags_state"] = fs
        reasons = []
        status = m.get("status")
        if status != "complete" and not args.include_partial:
            reasons.append(f"status={status!r} (not complete)")
        if dig(m, "flags", "exploratory", default=False) and not args.include_exploratory:
            reasons.append("flags.exploratory=true")
        if fs["nonconformant"] and not args.include_nonconformant:
            reasons.append("flags.nonconformant=true" + _reasons_suffix(fs["nonconformant_reasons"]))
        if dig(m, "flags", "truncated", default=False) and not args.include_truncated:
            reasons.append("flags.truncated=true")
        if reasons:
            run["exclusion_reasons"] = reasons
            excluded.append(run)
            continue
        included.append(run)
        if fs["nonconformant"]:  # only reachable with --include-nonconformant
            diag.warn(
                "nonconformant_included",
                f"{run['run_id']}: flags.nonconformant=true but included via "
                "--include-nonconformant — this run's harness deviated, the comparison is not "
                "like-for-like" + _reasons_suffix(fs["nonconformant_reasons"]),
            )
        if fs["provenance_incomplete"]:
            diag.warn(
                "provenance_incomplete",
                f"{run['run_id']}: flags.provenance_incomplete=true — included (the science is "
                "intact) but its cost/provenance attribution is APPROXIMATE. Unresolved: "
                + (", ".join(fs["provenance_incomplete_reasons"]) or "(no reasons recorded)"),
            )
    return included, excluded


def provenance_notes(runs: list[dict]) -> list[dict]:
    """One row per included run whose provenance is incomplete, for the report + stderr note."""
    rows = []
    for run in runs:
        fs = run.get("flags_state") or flag_state(run["manifest"])
        if not fs["provenance_incomplete"]:
            continue
        rows.append(
            {
                "run_id": run["run_id"],
                "model": dig(run["manifest"], "model", "name", default="?"),
                "suite": dig(run["manifest"], "suite", "name", default="?"),
                "unresolved": fs["provenance_incomplete_reasons"],
                "price_source": dig(run["manifest"], "price", "source", default=None),
                "lambda_instance_id": dig(
                    run["manifest"], "hardware", "lambda_instance_id", default=None
                ),
                "region": dig(run["manifest"], "hardware", "region", default=None),
            }
        )
    return rows


# ---------------------------------------------------------------- comparability


# Every knob that can change a verdict is BLOCKING: mixing it needs an explicit --allow-mixed
# and is annotated everywhere. "The harness was identical" is the study's central claim, so a
# silent warning is not an acceptable way to report that it was not (CONTRACTS.md §0.1).
#
# (label, manifest path, blocking) — compared across ALL included runs.
COMPARABILITY_KEYS = (
    ("harness.version", ("harness", "version"), True),
    ("harness.prompt_dir_sha256", ("harness", "prompt_dir_sha256"), True),
    ("harness.prompt_template_id", ("harness", "prompt_template_id"), True),
    ("harness.agent_config_sha256", ("harness", "agent_config_sha256"), True),
    # written by harness/manifest.py once the adapters-dir digest lands; absent on older
    # manifests, which is reported as drift rather than as a violation (see below).
    ("harness.adapters_dir_sha256", ("harness", "adapters_dir_sha256"), True),
    ("inference.temperature", ("inference", "temperature"), True),
    ("inference.top_p", ("inference", "top_p"), True),
    ("inference.top_k", ("inference", "top_k"), True),
    ("inference.seed", ("inference", "seed"), True),
    ("inference.max_iters", ("inference", "max_iters"), True),
    ("inference.max_tokens", ("inference", "max_tokens"), True),
    ("inference.max_attempt_tokens", ("inference", "max_attempt_tokens"), True),
    # --task-timeout is the BUDGET_WALLCLOCK ceiling (§1.2), so it decides verdicts.
    ("inference.task_timeout_s", ("inference", "task_timeout_s"), True),
    ("runtime.max_model_len", ("runtime", "max_model_len"), True),
    # Per-model EXTRA_ARGS reach the vLLM command line (modelctl). The held-constant flags
    # win under argparse last-wins, but any other serving flag (quantisation, KV cache dtype,
    # reasoning parsers, ...) can change what the model emits, so it may not vary silently.
    # None and "" are the same value: "no extra args" (see _COMPARABILITY_NORMALISERS).
    ("runtime.extra_args", ("runtime", "extra_args"), True),
    ("harness.result_schema", ("harness", "result_schema"), False),
    # §1.2: concurrency affects throughput and latency percentiles, NOT verdicts.
    ("inference.concurrency", ("inference", "concurrency"), False),
    ("inference.passes", ("inference", "passes"), False),
)

# Compared WITHIN each suite: the adapter and its grading environment are legitimately
# different between suites, so a global comparison would be meaningless.
PER_SUITE_KEYS = (
    ("harness.adapter_version", ("harness", "adapter_version"), True),
    # written by the adapters once environment_digest() is recorded in the manifest.
    ("harness.environment_digest", ("harness", "environment_digest"), True),
    ("harness.adapter_sha256", ("harness", "adapter_sha256"), False),
    ("suite.instance_ids_sha256", ("suite", "instance_ids_sha256"), False),
)


_MISSING = object()


def _field_values(runs: list[dict], path, normalise=None):
    """Distinct non-null values for a manifest path, plus the runs that do not record it."""
    values: list = []
    absent: list = []
    for run in runs:
        # Walk by hand rather than via dig(): dig() folds an explicit null into "absent",
        # but a normaliser may want to see it (runtime.extra_args: null == "" == no args).
        val = run["manifest"]
        for key in path:
            if not isinstance(val, dict) or key not in val:
                val = _MISSING
                break
            val = val[key]
        if val is _MISSING:
            absent.append(run["run_id"])
            continue
        if normalise is not None:
            val = normalise(val)
        if val is None:
            absent.append(run["run_id"])
            continue
        if val not in values:
            values.append(val)
    return values, absent


def _normalise_argstr(val):
    """A serving-flag string: None, "" and whitespace-only all mean "no extra args"."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        val = " ".join(str(v) for v in val)
    return " ".join(str(val).split())


# label → normaliser applied before values are compared (identity when absent).
_COMPARABILITY_NORMALISERS = {
    "runtime.extra_args": _normalise_argstr,
}


def comparability_report(runs: list[dict], args, diag: Diagnostics) -> dict:
    """The study's central claim is that the harness is constant. Enforce it here."""
    blocking: list = []
    soft: list = []
    fields: list = []

    def examine(label, path, is_blocking, scope, subset):
        values, absent = _field_values(subset, path, _COMPARABILITY_NORMALISERS.get(label))
        entry = {
            "field": label,
            "scope": scope,
            "blocking": bool(is_blocking),
            "values": values,
            "not_recorded_by": absent,
        }
        fields.append(entry)
        where = "" if scope == "all runs" else f" within {scope}"
        if len(values) > 1:
            msg = f"{label} differs{where} across runs: {values}"
            (blocking if is_blocking else soft).append(msg)
        elif absent and values:
            # Some manifests predate the field. Never a violation on its own — but say so,
            # because an unrecorded knob is an unverified knob.
            soft.append(
                f"{label} is not recorded by {len(absent)} run(s){where} "
                f"({', '.join(absent[:3])}{'…' if len(absent) > 3 else ''}) — "
                "the constant could not be verified for them"
            )
        elif absent and not values:
            soft.append(
                f"{label} is not recorded by any run{where} — the constant could not be verified"
            )
        return entry

    for label, path, is_blocking in COMPARABILITY_KEYS:
        examine(label, path, is_blocking, "all runs", runs)

    by_suite: dict[str, list] = {}
    for run in runs:
        by_suite.setdefault(dig(run["manifest"], "suite", "name", default="?"), []).append(run)
    for suite in sorted(by_suite):
        for label, path, is_blocking in PER_SUITE_KEYS:
            examine(label, path, is_blocking, f"suite {suite}", by_suite[suite])

    # Legacy views, kept because summary.json consumers and render_markdown read them.
    def vals(label):
        for entry in fields:
            if entry["field"] == label and entry["scope"] == "all runs":
                return entry["values"]
        return []

    adapters: dict[str, list] = {}
    for suite, suite_runs in by_suite.items():
        adapters[suite] = _field_values(suite_runs, ("harness", "adapter_version"))[0]

    mixed = bool(blocking)
    for msg in blocking:
        if args.allow_mixed:
            diag.warn("harness_mixed", "COMPARABILITY VIOLATION (allowed by --allow-mixed): " + msg)
        else:
            diag.error("harness_mixed", "COMPARABILITY VIOLATION: " + msg)
    for msg in soft:
        if args.strict and not args.allow_mixed:
            diag.error("comparability_soft", "COMPARABILITY DRIFT (--strict): " + msg)
        else:
            diag.warn("comparability_soft", "COMPARABILITY DRIFT: " + msg)

    return {
        "mixed": mixed,
        "allow_mixed": bool(args.allow_mixed),
        "blocking_differences": blocking,
        "soft_differences": soft,
        "fields": fields,
        "blocking_fields": [e["field"] for e in fields if e["blocking"]],
        "harness_versions": vals("harness.version"),
        "prompt_dir_sha256": vals("harness.prompt_dir_sha256"),
        "prompt_template_ids": vals("harness.prompt_template_id"),
        "agent_config_sha256": vals("harness.agent_config_sha256"),
        "adapters_dir_sha256": vals("harness.adapters_dir_sha256"),
        "result_schemas": vals("harness.result_schema"),
        "inference": {
            "temperature": vals("inference.temperature"),
            "top_p": vals("inference.top_p"),
            "top_k": vals("inference.top_k"),
            "seed": vals("inference.seed"),
            "max_iters": vals("inference.max_iters"),
            "max_tokens": vals("inference.max_tokens"),
            "max_attempt_tokens": vals("inference.max_attempt_tokens"),
            "task_timeout_s": vals("inference.task_timeout_s"),
        },
        "runtime": {
            "max_model_len": vals("runtime.max_model_len"),
            "extra_args": vals("runtime.extra_args"),
        },
        "adapter_versions_by_suite": adapters,
    }


# ---------------------------------------------------------------- metrics


def is_infra(code: str) -> bool:
    return code.startswith("INFRA_")


def compute_group(model: str, suite: str, runs: list[dict], setup_costs: dict[str, float],
                  api_pricing: dict, args, diag: Diagnostics) -> dict:
    records: list[dict] = []
    for run in runs:
        records.extend(run["records"])

    attempts_total = len(records)
    scored = [r for r in records if not is_infra(r["error_code"])]
    infra = [r for r in records if is_infra(r["error_code"])]
    resolved = [r for r in scored if r.get("resolved")]
    attempts_scored = len(scored)
    resolved_attempts = len(resolved)
    resolve_rate = (resolved_attempts / attempts_scored) if attempts_scored else None

    # ---- per-pass rates (a "pass" is one (run_id, pass_idx) slice)
    pass_slices: dict[tuple[str, int], dict[str, int]] = {}
    for r in records:
        key = (r["run_id"], r["pass_idx"])
        slot = pass_slices.setdefault(key, {"scored": 0, "resolved": 0, "infra": 0})
        if is_infra(r["error_code"]):
            slot["infra"] += 1
        else:
            slot["scored"] += 1
            if r.get("resolved"):
                slot["resolved"] += 1
    pass_rows = []
    for (run_id, pass_idx) in sorted(pass_slices):
        slot = pass_slices[(run_id, pass_idx)]
        rate = (slot["resolved"] / slot["scored"]) if slot["scored"] else None
        pass_rows.append(
            {
                "run_id": run_id,
                "pass_idx": pass_idx,
                "scored": slot["scored"],
                "resolved": slot["resolved"],
                "infra_excluded": slot["infra"],
                "resolve_rate": rate,
            }
        )
    pass_rates = [row["resolve_rate"] for row in pass_rows if row["resolve_rate"] is not None]

    # ---- per-instance rollup (pass@k and the cluster bootstrap)
    per_instance: dict[str, dict[str, int]] = {}
    for r in scored:
        slot = per_instance.setdefault(r["instance_id"], {"scored": 0, "resolved": 0})
        slot["scored"] += 1
        if r.get("resolved"):
            slot["resolved"] += 1
    instances_scored = len(per_instance)
    any_pass_resolved = sum(1 for v in per_instance.values() if v["resolved"] > 0)
    all_pass_resolved = sum(
        1 for v in per_instance.values() if v["scored"] > 0 and v["resolved"] == v["scored"]
    )
    pass_at_k = (any_pass_resolved / instances_scored) if instances_scored else None
    all_at_k = (all_pass_resolved / instances_scored) if instances_scored else None

    # The passes share one seed at temperature 0.0 (greedy decoding — see
    # harness/agent.py attempt_seed), so they are NOT independent samples and a bootstrap
    # over them would report a sampling CI the design cannot support. Report the observed
    # spread instead: mean + min/max, which is what the study methodology specifies.
    # The INSTANCE-level cluster bootstrap below stays valid — it resamples instances,
    # which genuinely were sampled.
    ci_low, ci_high = (min(pass_rates), max(pass_rates)) if pass_rates else (None, None)
    ci_lo_i, ci_hi_i = bootstrap_ci_over_instances(
        [(v["resolved"], v["scored"]) for v in per_instance.values()], args.bootstrap_iters
    )

    # ---- cost
    billing_modes = sorted({billing_mode_for(run["manifest"], api_pricing) for run in runs})
    billing_mode = billing_modes[0] if len(billing_modes) == 1 else "mixed"
    if billing_mode == "mixed":
        diag.warn(
            "billing_mode_mixed",
            f"{model}/{suite}: runs use different billing modes {billing_modes}; costs summed anyway",
        )

    gpu_hours = 0.0
    hours_cost_usd = 0.0
    cost_known = True
    cost_sources: list[str] = []
    price_cph_values: list[float] = []
    token_cost_detail = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "input_usd": 0.0,
        "cached_input_usd": 0.0,
        "output_usd": 0.0,
    }

    for run in runs:
        m = run["manifest"]
        mode = billing_mode_for(m, api_pricing)
        if mode == "per_token":
            rates, src = api_rates_for(m, api_pricing)
            if not rates or rates.get("input_usd_per_mtok") is None:
                diag.error(
                    "api_pricing_unavailable",
                    f"{run['run_id']}: model is API-baseline (per-token billing) but no rates "
                    "were found — pass --api-pricing with an entry for this model",
                )
                cost_known = False
                continue
            r_in = float(rates.get("input_usd_per_mtok") or 0.0)
            r_out = float(rates.get("output_usd_per_mtok") or 0.0)
            r_cached = rates.get("cached_input_usd_per_mtok")
            r_cached = float(r_cached) if isinstance(r_cached, (int, float)) else r_in
            for r in run["records"]:
                prompt = int(dig(r, "tokens", "prompt", default=0) or 0)
                cached = int(dig(r, "tokens", "cached_prompt", default=0) or 0)
                completion = int(dig(r, "tokens", "completion", default=0) or 0)
                cached = min(cached, prompt)
                fresh = prompt - cached
                token_cost_detail["input_tokens"] += fresh
                token_cost_detail["cached_input_tokens"] += cached
                token_cost_detail["output_tokens"] += completion
                token_cost_detail["input_usd"] += fresh / 1e6 * r_in
                token_cost_detail["cached_input_usd"] += cached / 1e6 * r_cached
                token_cost_detail["output_usd"] += completion / 1e6 * r_out
            if f"per-token:{src}" not in cost_sources:
                cost_sources.append(f"per-token:{src}")
        else:
            # active_wall_clock_s excludes the idle gap between a --resume's invocations
            # (CONTRACTS.md §2.2/§8/§9): billing that gap would make the headline metric
            # depend on when an operator happened to type --resume. Fall back only for
            # pre-split manifests, and say so.
            wall_s = dig(m, "timing", "active_wall_clock_s", default=None)
            if not isinstance(wall_s, (int, float)):
                wall_s = dig(m, "timing", "wall_clock_s", default=None)
                if isinstance(wall_s, (int, float)):
                    diag.warn(
                        "timing_active_wall_clock_missing",
                        f"{run['run_id']}: no timing.active_wall_clock_s (pre-split manifest); "
                        "cost uses idle-inclusive wall_clock_s",
                    )
            cph = run_price_cph(m, diag)
            if isinstance(cph, (int, float)):
                price_cph_values.append(float(cph))
            if isinstance(wall_s, (int, float)) and isinstance(cph, (int, float)):
                hours = float(wall_s) / 3600.0
                gpu_hours += hours
                hours_cost_usd += hours * float(cph) / 100.0
                if "instance-hours:manifest" not in cost_sources:
                    cost_sources.append("instance-hours:manifest")
            else:
                attributable = sum(
                    float(dig(r, "cost", "usd", default=0.0) or 0.0) for r in run["records"]
                )
                if attributable > 0:
                    hours_cost_usd += attributable
                    gpu_hours += sum(
                        float(dig(r, "cost", "gpu_seconds", default=0.0) or 0.0)
                        for r in run["records"]
                    ) / 3600.0
                    diag.warn(
                        "cost_attributed_fallback",
                        f"{run['run_id']}: manifest timing.wall_clock_s or price missing; fell "
                        "back to summing per-attempt cost.usd (under-counts idle time)",
                    )
                    if "instance-hours:attributed-fallback" not in cost_sources:
                        cost_sources.append("instance-hours:attributed-fallback")
                else:
                    fs = run.get("flags_state") or flag_state(m)
                    if fs["provenance_incomplete"]:
                        # §9: a provenance_incomplete run is INCLUDED — its science is intact,
                        # only the dollar columns are unknowable. Blank them ("—"), mark the
                        # group approximate, and say so; refusing the whole aggregation here
                        # would throw away scored attempts over a missing price.
                        diag.warn(
                            "cost_unknown",
                            f"{run['run_id']}: provenance_incomplete run has no wall_clock_s/price "
                            "and no per-attempt cost — included, but its group's cost columns "
                            "are left blank",
                        )
                        if "instance-hours:unknown" not in cost_sources:
                            cost_sources.append("instance-hours:unknown")
                    else:
                        diag.error(
                            "cost_unresolvable",
                            f"{run['run_id']}: no wall_clock_s/price and no per-attempt cost — "
                            "cost cannot be computed for this run",
                        )
                    cost_known = False

    token_cost_usd = (
        token_cost_detail["input_usd"]
        + token_cost_detail["cached_input_usd"]
        + token_cost_detail["output_usd"]
    )
    cost_usd = hours_cost_usd + token_cost_usd

    # ---- cost-attribution quality. flags.provenance_incomplete does not exclude a run, but
    # every cost number derived from it is approximate and must be labelled as such (§8).
    provenance_runs = []
    for run in runs:
        fs = run.get("flags_state") or flag_state(run["manifest"])
        if fs["provenance_incomplete"]:
            provenance_runs.append(
                {
                    "run_id": run["run_id"],
                    "unresolved": fs["provenance_incomplete_reasons"] or ["(no reasons recorded)"],
                }
            )
    cost_approximate_reasons = [
        f"{p['run_id']}: " + ", ".join(p["unresolved"]) for p in provenance_runs
    ]
    if "instance-hours:attributed-fallback" in cost_sources:
        cost_approximate_reasons.append(
            "cost fell back to summing per-attempt cost.usd for at least one run "
            "(under-counts idle instance time)"
        )
    cost_approximate = bool(cost_approximate_reasons)
    if cost_approximate:
        diag.warn(
            "cost_approximate",
            f"{model}/{suite}: cost columns are APPROXIMATE — "
            + "; ".join(cost_approximate_reasons),
        )

    attributable_cost = sum(float(dig(r, "cost", "usd", default=0.0) or 0.0) for r in records)
    cost_per_resolved = (
        (cost_usd / resolved_attempts) if (cost_known and resolved_attempts > 0) else None
    )
    cost_per_attempt = (cost_usd / attempts_total) if (cost_known and attempts_total) else None
    setup_cost = sum(setup_costs.get(run["run_id"], 0.0) for run in runs) or None

    # ---- latency
    wall_s = sorted(
        float(r["wall_clock_ms"]) / 1000.0 for r in records if isinstance(r.get("wall_clock_ms"), (int, float))
    )
    gen_s = sorted(
        float(dig(r, "latency_ms", "generation_total", default=0) or 0) / 1000.0
        for r in records
        if dig(r, "latency_ms", "generation_total", default=None) is not None
    )
    ttft = [
        float(dig(r, "latency_ms", "ttft_p50", default=0) or 0)
        for r in records
        if dig(r, "latency_ms", "ttft_p50", default=None) is not None
    ]
    per_call = [
        float(dig(r, "latency_ms", "per_call_p50", default=0) or 0)
        for r in records
        if dig(r, "latency_ms", "per_call_p50", default=None) is not None
    ]

    # ---- tokens
    tok_prompt = [int(dig(r, "tokens", "prompt", default=0) or 0) for r in records]
    tok_completion = [int(dig(r, "tokens", "completion", default=0) or 0) for r in records]
    tok_total = [
        int(dig(r, "tokens", "total", default=0) or 0)
        or (int(dig(r, "tokens", "prompt", default=0) or 0) + int(dig(r, "tokens", "completion", default=0) or 0))
        for r in records
    ]
    tok_cached = [int(dig(r, "tokens", "cached_prompt", default=0) or 0) for r in records]
    iters = [int(r["iterations"]) for r in records if isinstance(r.get("iterations"), int)]

    # ---- failure taxonomy
    counts = {code: 0 for code in ERROR_CODES}
    for r in records:
        counts[r["error_code"]] += 1
    infra_unknown_share = (counts["INFRA_UNKNOWN"] / attempts_total) if attempts_total else 0.0
    infra_grader_share = (counts["INFRA_GRADER"] / attempts_total) if attempts_total else 0.0
    server_unavailable_share = (
        (counts["SERVER_UNAVAILABLE"] / attempts_scored) if attempts_scored else 0.0
    )
    if infra_unknown_share > INFRA_UNKNOWN_INVALID_SHARE:
        diag.warn(
            "infra_unknown_over_threshold",
            f"{model}/{suite}: INFRA_UNKNOWN is {infra_unknown_share:.1%} of attempts (>2%) — "
            "per §4 these runs are INVALID and must be re-run",
        )
    if infra_grader_share > GRADER_DEGRADED_SHARE:
        diag.warn(
            "grading_degraded",
            f"{model}/{suite}: INFRA_GRADER is {infra_grader_share:.1%} of attempts (>2%) — "
            "grading degraded",
        )
    if server_unavailable_share > SERVER_UNAVAILABLE_FLAG_SHARE:
        diag.warn(
            "server_unavailable_high",
            f"{model}/{suite}: SERVER_UNAVAILABLE is {server_unavailable_share:.1%} of scored "
            "attempts (>5%) — must be flagged in the writeup",
        )

    return {
        "model": model,
        "suite": suite,
        "runs": len(runs),
        "run_ids": [run["run_id"] for run in runs],
        "statuses": sorted({run["manifest"].get("status") for run in runs}),
        "passes_observed": len(pass_rows),
        "instances_scored": instances_scored,
        "attempts_total": attempts_total,
        "attempts_scored": attempts_scored,
        "attempts_infra_excluded": len(infra),
        "resolved_attempts": resolved_attempts,
        "resolve_rate": resolve_rate,
        "resolve_rate_pass_min": ci_low,
        "resolve_rate_pass_max": ci_high,
        "resolve_rate_ci95_low_instance_bootstrap": ci_lo_i,
        "resolve_rate_ci95_high_instance_bootstrap": ci_hi_i,
        "pass_rate_mean": mean_or_none(pass_rates),
        "pass_rate_min": min(pass_rates) if pass_rates else None,
        "pass_rate_max": max(pass_rates) if pass_rates else None,
        "pass_rates": pass_rows,
        "pass_at_1": mean_or_none(pass_rates),
        "pass_at_k": pass_at_k,
        "all_passes_resolved_rate": all_at_k,
        "k": len(pass_rows),
        "billing_mode": billing_mode,
        "cost_sources": cost_sources,
        "cost_known": cost_known,
        "cost_approximate": cost_approximate,
        "cost_approximate_reasons": cost_approximate_reasons,
        "provenance_incomplete_runs": provenance_runs,
        # A partial hours total (some run's wall clock unknown) would read as a real number.
        "gpu_hours": gpu_hours if (billing_mode != "per_token" and cost_known) else None,
        "effective_cents_per_hour": sorted(set(price_cph_values)) or None,
        "cost_usd": cost_usd if cost_known else None,
        "attributable_cost_usd": attributable_cost or None,
        "cost_per_resolved_usd": cost_per_resolved,
        "cost_per_attempt_usd": cost_per_attempt,
        "setup_cost_usd": setup_cost,
        "token_cost_detail": token_cost_detail if billing_mode in ("per_token", "mixed") else None,
        "latency": {
            "attempt_wall_s_p50": percentile(wall_s, 0.50),
            "attempt_wall_s_p95": percentile(wall_s, 0.95),
            "generation_s_p50": percentile(gen_s, 0.50),
            "generation_s_p95": percentile(gen_s, 0.95),
            "ttft_ms_median_of_attempt_p50": median_or_none(ttft),
            "per_call_ms_median_of_attempt_p50": median_or_none(per_call),
        },
        "tokens": {
            "prompt_median": median_or_none(tok_prompt),
            "completion_median": median_or_none(tok_completion),
            "total_median": median_or_none(tok_total),
            "total_mean": mean_or_none(tok_total),
            "prompt_sum": sum(tok_prompt),
            "completion_sum": sum(tok_completion),
            "total_sum": sum(tok_total),
            "cached_prompt_sum": sum(tok_cached),
        },
        "iterations_median": median_or_none(iters),
        "failure_counts": counts,
        "infra_unknown_share": infra_unknown_share,
        "infra_grader_share": infra_grader_share,
        "server_unavailable_share": server_unavailable_share,
        "run_flags": {
            "grading_degraded_any": any(
                dig(run["manifest"], "flags", "grading_degraded", default=False) for run in runs
            ),
            "nonconformant_any": any(
                dig(run["manifest"], "flags", "nonconformant", default=False) for run in runs
            ),
            "provenance_incomplete_any": bool(provenance_runs),
            "consent_classes": sorted(
                {dig(run["manifest"], "flags", "consent_class", default="unknown") for run in runs}
            ),
        },
    }


def rollup_by_model(groups: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for g in groups:
        by_model.setdefault(g["model"], []).append(g)
    rows = []
    for model, gs in sorted(by_model.items()):
        cost_known = all(g["cost_known"] for g in gs)
        cost = sum(g["cost_usd"] or 0.0 for g in gs) if cost_known else None
        resolved = sum(g["resolved_attempts"] for g in gs)
        scored = sum(g["attempts_scored"] for g in gs)
        rows.append(
            {
                "model": model,
                "suites": [g["suite"] for g in gs],
                "attempts_total": sum(g["attempts_total"] for g in gs),
                "attempts_scored": scored,
                "resolved_attempts": resolved,
                "resolve_rate": (resolved / scored) if scored else None,
                "cost_usd": cost,
                "cost_per_resolved_usd": (cost / resolved) if (cost is not None and resolved) else None,
                "setup_cost_usd": sum(g["setup_cost_usd"] or 0.0 for g in gs) or None,
                "gpu_hours": sum(g["gpu_hours"] or 0.0 for g in gs) or None,
                "billing_modes": sorted({g["billing_mode"] for g in gs}),
                "cost_approximate": any(g["cost_approximate"] for g in gs),
            }
        )
    return rows


def contamination_view(groups: list[dict], threshold: float) -> list[dict]:
    """Per-model Verified vs Pro vs AgentTask deltas — the study's contamination evidence."""
    by_model: dict[str, dict[str, dict]] = {}
    for g in groups:
        by_model.setdefault(g["model"], {})[g["suite"]] = g
    rows = []
    for model, per_suite in sorted(by_model.items()):
        rates = {s: (per_suite[s]["resolve_rate"] if s in per_suite else None) for s in SUITE_ORDER}
        ver = rates["swebench-verified"]
        pro = rates["swebench-pro"]
        agt = rates["agenttask"]
        others = [v for k, v in rates.items() if k != "swebench-verified" and v is not None]
        gap = (ver - _mean(others)) if (ver is not None and others) else None
        rows.append(
            {
                "model": model,
                "verified_resolve_rate": ver,
                "pro_resolve_rate": pro,
                "agenttask_resolve_rate": agt,
                "delta_verified_minus_pro": (ver - pro) if (ver is not None and pro is not None) else None,
                "delta_verified_minus_agenttask": (ver - agt) if (ver is not None and agt is not None) else None,
                "delta_pro_minus_agenttask": (pro - agt) if (pro is not None and agt is not None) else None,
                "verified_gap_vs_mean_others": gap,
                "contamination_flag": bool(gap is not None and gap >= threshold),
                "suites_present": sorted(per_suite.keys()),
                "complete_triple": all(rates[s] is not None for s in SUITE_ORDER),
            }
        )
    return rows


# ---------------------------------------------------------------- formatting


def fmt_pct(x, digits=1):
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def fmt_usd(x, digits=2):
    if x is None:
        return "—"
    if abs(x) < 0.01 and x != 0:
        return f"${x:.4f}"
    return f"${x:,.{digits}f}"


def fmt_num(x, digits=0):
    if x is None:
        return "—"
    if digits == 0:
        return f"{round(x):,}"
    return f"{x:,.{digits}f}"


def fmt_usd_approx(x, approximate: bool, digits=2):
    """Cost cell. A run flagged provenance_incomplete makes its group's costs approximate."""
    text = fmt_usd(x, digits)
    if approximate and text != "—":
        return "≈" + text
    return text


def fmt_delta_pct(x, digits=1):
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.{digits}f} pp"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(out) + "\n"


def render_markdown(report: dict) -> str:
    comp = report["comparability"]
    groups = report["by_model_suite"]
    out: list[str] = []
    a = out.append

    a("# The Harness Variable — aggregate results\n")
    a(
        f"Generated {report['generated_at']} by `analysis/aggregate.py` "
        f"v{report['aggregator_version']} · report schema `{report['schema']}`\n"
    )

    if comp["mixed"]:
        a("> ## MIXED-HARNESS AGGREGATE — NOT PUBLISHABLE AS A LIKE-FOR-LIKE COMPARISON\n>")
        a(
            "> These runs do **not** share one harness. The study's central claim is that the "
            "harness is the control variable; it does not hold for this table. Every knob below "
            "parameterises a verdict, so a difference in any one of them makes the numbers "
            "incomparable — not merely noisy.\n>"
        )
        for msg in comp["blocking_differences"]:
            a(f"> - {msg}")
        a(">\n> Produced only because `--allow-mixed` was passed.\n")

    a("## Harness constants\n")
    a(
        "Every row marked **blocking** is a verdict-affecting knob: if it is not a single value "
        "across the included runs, `aggregate.py` refuses to aggregate without `--allow-mixed`.\n"
    )

    def _short(value) -> str:
        """Abbreviate digests so the table stays readable; leave everything else alone."""
        text = str(value)
        prefix = "sha256:" if text.startswith("sha256:") else ""
        body = text[len(prefix):]
        if len(body) == 64 and all(c in "0123456789abcdef" for c in body.lower()):
            return prefix + body[:16] + "…"
        return text

    const_rows = []
    for entry in comp.get("fields", []):
        values = ", ".join(_short(v) for v in entry["values"]) or "—"
        note = ""
        if entry["not_recorded_by"]:
            note = f" _(not recorded by {len(entry['not_recorded_by'])} run(s))_"
        const_rows.append(
            [
                entry["field"],
                entry["scope"],
                "blocking" if entry["blocking"] else "soft",
                values + note,
            ]
        )
    a(md_table(["constant", "compared over", "mixing", "value(s)"], const_rows))

    if comp["soft_differences"]:
        a("**Comparability drift detected (non-blocking):**\n")
        for msg in comp["soft_differences"]:
            a(f"- {msg}")
        a("")

    # ---- headline
    a("## Headline — cost per resolved task\n")
    a(
        "`cost_per_resolved = cost_usd / resolved_attempts`, where `cost_usd` is instance-hours "
        "× the manifest price snapshot (or per-token API pricing for baseline models). Setup and "
        "weight-download time is **excluded** and reported separately (CONTRACTS.md §8).\n"
    )
    headline_rows = []
    for g in sorted(
        groups,
        key=lambda g: (
            g["cost_per_resolved_usd"] is None,
            g["cost_per_resolved_usd"] if g["cost_per_resolved_usd"] is not None else 0.0,
        ),
    ):
        approx = g["cost_approximate"]
        headline_rows.append(
            [
                g["model"],
                SUITE_SHORT.get(g["suite"], g["suite"]),
                g["billing_mode"],
                fmt_pct(g["resolve_rate"]),
                str(g["resolved_attempts"]),
                str(g["attempts_scored"]),
                fmt_num(g["gpu_hours"], 2) if g["gpu_hours"] is not None else "—",
                fmt_usd_approx(g["cost_usd"], approx),
                f"**{fmt_usd_approx(g['cost_per_resolved_usd'], approx)}**",
                fmt_usd_approx(g["setup_cost_usd"], approx),
            ]
        )
    a(
        md_table(
            [
                "model",
                "suite",
                "billing",
                "resolve rate",
                "resolved",
                "scored",
                "GPU h",
                "cost",
                "cost / resolved",
                "setup (sep.)",
            ],
            headline_rows,
        )
    )

    prov = report.get("provenance_incomplete_runs") or []
    if prov:
        a(
            "**≈ marks approximate cost.** Those groups include at least one run flagged "
            "`flags.provenance_incomplete`: the run itself is scientifically sound and its "
            "resolve rate is exact, but part of its cost/provenance attribution could not be "
            "resolved, so the dollar columns are best-effort. Resolve rates, token counts and "
            "the failure taxonomy are **not** affected.\n"
        )
        a("### Provenance-incomplete runs (included; cost approximate)\n")
        a(
            md_table(
                ["run_id", "model", "suite", "unresolved provenance fields"],
                [
                    [
                        p["run_id"],
                        p["model"],
                        SUITE_SHORT.get(p["suite"], p["suite"]),
                        "; ".join(p["unresolved"]) or "(no reasons recorded)",
                    ]
                    for p in prov
                ],
            )
        )
        a(
            "_These runs are included by design: `flags.provenance_incomplete` means the cost "
            "attribution is imprecise, not that the harness deviated. A run whose harness "
            "actually deviated carries `flags.nonconformant` and is excluded — see \"Runs "
            "excluded\"._\n"
        )

    # ---- resolution detail
    a("## Resolution rate — mean and range over passes\n")
    res_rows = []
    for g in sorted(groups, key=lambda g: (SUITE_ORDER.index(g["suite"]) if g["suite"] in SUITE_ORDER else 9, g["model"])):
        ci = (
            f"[{fmt_pct(g['resolve_rate_pass_min'])}, {fmt_pct(g['resolve_rate_pass_max'])}]"
            if g["resolve_rate_pass_min"] is not None
            else "—"
        )
        rng = (
            f"{fmt_pct(g['pass_rate_min'])} – {fmt_pct(g['pass_rate_max'])}"
            if g["pass_rate_min"] is not None
            else "—"
        )
        res_rows.append(
            [
                g["model"],
                SUITE_SHORT.get(g["suite"], g["suite"]),
                str(g["k"]),
                fmt_pct(g["resolve_rate"]),
                fmt_pct(g["pass_rate_mean"]),
                rng,
                ci,
                fmt_pct(g["pass_at_k"]),
                fmt_pct(g["all_passes_resolved_rate"]),
                str(g["instances_scored"]),
            ]
        )
    a(
        md_table(
            [
                "model",
                "suite",
                "passes",
                "pooled",
                "pass@1 (mean)",
                "pass range",
                "range over passes",
                "pass@k (any)",
                "all passes",
                "instances",
            ],
            res_rows,
        )
    )
    a(
        "_Pooled rate uses the §4 denominator: `INFRA_*` attempts are excluded, `SERVER_*` "
        "attempts are **kept** (an unservable model is a real cost of that model)._\n"
    )

    # ---- latency + tokens
    a("## Latency and tokens per task\n")
    lt_rows = []
    for g in sorted(groups, key=lambda g: (SUITE_ORDER.index(g["suite"]) if g["suite"] in SUITE_ORDER else 9, g["model"])):
        lat = g["latency"]
        tok = g["tokens"]
        lt_rows.append(
            [
                g["model"],
                SUITE_SHORT.get(g["suite"], g["suite"]),
                fmt_num(lat["attempt_wall_s_p50"], 1),
                fmt_num(lat["attempt_wall_s_p95"], 1),
                fmt_num(lat["generation_s_p50"], 1),
                fmt_num(lat["generation_s_p95"], 1),
                fmt_num(lat["ttft_ms_median_of_attempt_p50"], 0),
                fmt_num(tok["prompt_median"]),
                fmt_num(tok["completion_median"]),
                fmt_num(tok["total_median"]),
                fmt_num(g["iterations_median"], 1),
            ]
        )
    a(
        md_table(
            [
                "model",
                "suite",
                "wall p50 (s)",
                "wall p95 (s)",
                "gen p50 (s)",
                "gen p95 (s)",
                "TTFT p50 (ms)",
                "tok in (med)",
                "tok out (med)",
                "tok total (med)",
                "iters (med)",
            ],
            lt_rows,
        )
    )

    # ---- failures
    a("## Failure taxonomy (CONTRACTS.md §4)\n")
    fam_rows = []
    for g in sorted(groups, key=lambda g: (SUITE_ORDER.index(g["suite"]) if g["suite"] in SUITE_ORDER else 9, g["model"])):
        counts = g["failure_counts"]
        row = [g["model"], SUITE_SHORT.get(g["suite"], g["suite"])]
        for name, codes in FAMILIES:
            row.append(str(sum(counts[c] for c in codes)))
        row.append(str(g["attempts_total"]))
        fam_rows.append(row)
    a(
        md_table(
            ["model", "suite"] + [name for name, _ in FAMILIES] + ["attempts"],
            fam_rows,
        )
    )
    a("_`infra*` attempts are excluded from every denominator; all other columns are included._\n")

    for suite in SUITE_ORDER:
        suite_groups = [g for g in groups if g["suite"] == suite]
        if not suite_groups:
            continue
        codes_present = [
            c for c in ERROR_CODES if any(g["failure_counts"][c] for g in suite_groups)
        ]
        if not codes_present:
            continue
        a(f"### {SUITE_SHORT.get(suite, suite)} — full code breakdown\n")
        rows = []
        for g in sorted(suite_groups, key=lambda g: g["model"]):
            rows.append([g["model"]] + [str(g["failure_counts"][c]) for c in codes_present])
        a(md_table(["model"] + codes_present, rows))

    # ---- contamination
    a("## Cross-suite contamination view\n")
    a(
        "Same model, same harness, three suites. A large **positive** Verified gap is the "
        "contamination signal: SWE-bench Verified predates these checkpoints and is plausibly in "
        "their pretraining data, while SWE-bench Pro and the internal AgentTask suite are not.\n"
    )
    cont_rows = []
    for row in report["contamination"]:
        cont_rows.append(
            [
                row["model"],
                fmt_pct(row["verified_resolve_rate"]),
                fmt_pct(row["pro_resolve_rate"]),
                fmt_pct(row["agenttask_resolve_rate"]),
                fmt_delta_pct(row["delta_verified_minus_pro"]),
                fmt_delta_pct(row["delta_verified_minus_agenttask"]),
                fmt_delta_pct(row["verified_gap_vs_mean_others"]),
                "**YES**" if row["contamination_flag"] else "no",
                "yes" if row["complete_triple"] else "no",
            ]
        )
    a(
        md_table(
            [
                "model",
                "Verified",
                "Pro",
                "AgentTask",
                "V − Pro",
                "V − AT",
                "V − mean(others)",
                f"flag (≥{report['options']['contamination_threshold'] * 100:.0f} pp)",
                "all 3 suites",
            ],
            cont_rows,
        )
    )
    a(
        "_Deltas are in percentage points of resolve rate. Rows without all three suites are "
        "shown for completeness but must not be quoted as contamination evidence._\n"
    )

    # ---- per-model rollup
    a("## Per-model rollup (all suites pooled)\n")
    a("_Suites have different task counts and difficulty; this rollup is a budget view, not a score._\n")
    roll_rows = []
    for row in report["by_model"]:
        roll_rows.append(
            [
                row["model"],
                ",".join(SUITE_SHORT.get(s, s) for s in row["suites"]),
                str(row["resolved_attempts"]),
                str(row["attempts_scored"]),
                fmt_pct(row["resolve_rate"]),
                fmt_num(row["gpu_hours"], 2),
                fmt_usd_approx(row["cost_usd"], row["cost_approximate"]),
                fmt_usd_approx(row["cost_per_resolved_usd"], row["cost_approximate"]),
                fmt_usd_approx(row["setup_cost_usd"], row["cost_approximate"]),
            ]
        )
    a(
        md_table(
            ["model", "suites", "resolved", "scored", "resolve rate", "GPU h", "cost",
             "cost / resolved", "setup (sep.)"],
            roll_rows,
        )
    )

    # ---- provenance
    a("## Runs included\n")
    run_rows = []
    for r in report["runs"]:
        run_rows.append(
            [
                r["run_id"],
                r["model"],
                SUITE_SHORT.get(r["suite"], r["suite"]),
                str(r["status"]),
                str(r["passes"]),
                str(r["records"]),
                fmt_num(r["wall_clock_h"], 2),
                str(r["effective_cents_per_hour"] if r["effective_cents_per_hour"] is not None else "—"),
                (r["weight_digest"] or "—")[:23],
                r["checksums"],
                "approx" if r["provenance_incomplete"] else "exact",
            ]
        )
    a(
        md_table(
            ["run_id", "model", "suite", "status", "passes", "records", "wall h", "¢/h",
             "weight digest", "checksums", "provenance"],
            run_rows,
        )
    )

    if report["excluded_runs"]:
        a("## Runs excluded\n")
        ex_rows = [
            [r["run_id"], r["model"], SUITE_SHORT.get(r["suite"], r["suite"]), "; ".join(r["exclusion_reasons"])]
            for r in report["excluded_runs"]
        ]
        a(md_table(["run_id", "model", "suite", "reason"], ex_rows))

    warnings = report["diagnostics"]["warnings"]
    if warnings:
        a("## Warnings\n")
        for w in warnings:
            a(f"- `{w['code']}` — {w['message']}")
        a("")

    a("---\n")
    a(
        f"Bootstrap: {report['options']['bootstrap_iters']} resamples, fixed seed "
        f"{BOOTSTRAP_SEED} (re-running reproduces identical intervals).\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------- CSV emitters


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def csv_by_model_suite(report: dict, path: Path) -> None:
    mixed = report["comparability"]["mixed"]
    headers = [
        "model", "suite", "runs", "passes", "instances_scored", "attempts_total",
        "attempts_scored", "attempts_infra_excluded", "resolved_attempts", "resolve_rate",
        "resolve_rate_pass_min", "resolve_rate_pass_max", "pass_rate_mean", "pass_rate_min",
        "pass_rate_max", "pass_at_1", "pass_at_k", "all_passes_resolved_rate", "billing_mode",
        "gpu_hours", "cost_usd", "cost_per_resolved_usd", "cost_per_attempt_usd",
        "setup_cost_usd", "cost_approximate", "cost_approximate_reasons",
        "attempt_wall_s_p50", "attempt_wall_s_p95", "generation_s_p50",
        "generation_s_p95", "tokens_prompt_median", "tokens_completion_median",
        "tokens_total_median", "tokens_total_sum", "iterations_median",
        "infra_unknown_share", "server_unavailable_share", "harness_mixed",
    ]
    rows = []
    for g in report["by_model_suite"]:
        rows.append(
            [
                g["model"], g["suite"], g["runs"], g["k"], g["instances_scored"],
                g["attempts_total"], g["attempts_scored"], g["attempts_infra_excluded"],
                g["resolved_attempts"], g["resolve_rate"], g["resolve_rate_pass_min"],
                g["resolve_rate_pass_max"], g["pass_rate_mean"], g["pass_rate_min"],
                g["pass_rate_max"], g["pass_at_1"], g["pass_at_k"], g["all_passes_resolved_rate"],
                g["billing_mode"], g["gpu_hours"], g["cost_usd"], g["cost_per_resolved_usd"],
                g["cost_per_attempt_usd"], g["setup_cost_usd"], g["cost_approximate"],
                "; ".join(g["cost_approximate_reasons"]),
                g["latency"]["attempt_wall_s_p50"], g["latency"]["attempt_wall_s_p95"],
                g["latency"]["generation_s_p50"], g["latency"]["generation_s_p95"],
                g["tokens"]["prompt_median"], g["tokens"]["completion_median"],
                g["tokens"]["total_median"], g["tokens"]["total_sum"], g["iterations_median"],
                g["infra_unknown_share"], g["server_unavailable_share"], mixed,
            ]
        )
    write_csv(path, headers, rows)


def csv_failures(report: dict, path: Path) -> None:
    mixed = report["comparability"]["mixed"]
    headers = [
        "model", "suite", "error_code", "count", "share_of_attempts", "in_denominator",
        "harness_mixed",
    ]
    rows = []
    for g in report["by_model_suite"]:
        total = g["attempts_total"] or 1
        for code in ERROR_CODES:
            rows.append(
                [
                    g["model"], g["suite"], code, g["failure_counts"][code],
                    g["failure_counts"][code] / total, (not is_infra(code)), mixed,
                ]
            )
    write_csv(path, headers, rows)


def csv_contamination(report: dict, path: Path) -> None:
    mixed = report["comparability"]["mixed"]
    headers = [
        "model", "verified_resolve_rate", "pro_resolve_rate", "agenttask_resolve_rate",
        "delta_verified_minus_pro", "delta_verified_minus_agenttask",
        "delta_pro_minus_agenttask", "verified_gap_vs_mean_others", "contamination_flag",
        "complete_triple", "harness_mixed",
    ]
    rows = []
    for r in report["contamination"]:
        rows.append(
            [
                r["model"], r["verified_resolve_rate"], r["pro_resolve_rate"],
                r["agenttask_resolve_rate"], r["delta_verified_minus_pro"],
                r["delta_verified_minus_agenttask"], r["delta_pro_minus_agenttask"],
                r["verified_gap_vs_mean_others"], r["contamination_flag"],
                r["complete_triple"], mixed,
            ]
        )
    write_csv(path, headers, rows)


def csv_runs(report: dict, path: Path) -> None:
    headers = [
        "run_id", "model", "suite", "status", "passes", "records", "wall_clock_s",
        "effective_cents_per_hour", "billing_mode", "harness_version", "prompt_dir_sha256",
        "adapter_version", "weight_revision", "weight_digest", "repo_git_sha", "instance_type",
        "region", "lambda_instance_id", "consent_class", "nonconformant",
        "nonconformant_reasons", "provenance_incomplete", "provenance_incomplete_reasons",
        "legacy_single_flag_manifest", "exploratory",
        "grading_degraded", "checksums", "included", "exclusion_reasons",
    ]
    rows = []
    for r in report["runs"] + report["excluded_runs"]:
        rows.append(
            [
                r["run_id"], r["model"], r["suite"], r["status"], r["passes"], r["records"],
                r["wall_clock_s"], r["effective_cents_per_hour"], r["billing_mode"],
                r["harness_version"], r["prompt_dir_sha256"], r["adapter_version"],
                r["weight_revision"], r["weight_digest"], r["repo_git_sha"], r["instance_type"],
                r["region"], r["lambda_instance_id"], r["consent_class"], r["nonconformant"],
                "; ".join(r["nonconformant_reasons"]), r["provenance_incomplete"],
                "; ".join(r["provenance_incomplete_reasons"]),
                r["legacy_single_flag_manifest"],
                r["exploratory"], r["grading_degraded"], r["checksums"], r["included"],
                "; ".join(r.get("exclusion_reasons", [])),
            ]
        )
    write_csv(path, headers, rows)


def run_summary_row(run: dict, api_pricing: dict, included: bool) -> dict:
    m = run["manifest"]
    fs = run.get("flags_state") or flag_state(m)
    wall = dig(m, "timing", "wall_clock_s", default=None)
    return {
        "run_id": run["run_id"],
        "model": dig(m, "model", "name", default="?"),
        "suite": dig(m, "suite", "name", default="?"),
        "status": m.get("status"),
        "passes": dig(m, "inference", "passes", default=None),
        "records": len(run["records"]),
        "wall_clock_s": wall,
        "wall_clock_h": (float(wall) / 3600.0) if isinstance(wall, (int, float)) else None,
        "effective_cents_per_hour": dig(m, "price", "effective_cents_per_hour", default=None),
        "price_source": dig(m, "price", "source", default=None),
        "billing_mode": billing_mode_for(m, api_pricing),
        "harness_version": dig(m, "harness", "version", default=None),
        "prompt_dir_sha256": dig(m, "harness", "prompt_dir_sha256", default=None),
        "adapter_version": dig(m, "harness", "adapter_version", default=None),
        "weight_revision": dig(m, "model", "weight_revision", default=None),
        "weight_digest": dig(m, "model", "weight_digest", default=None),
        "repo_git_sha": dig(m, "harness", "repo_git_sha", default=None),
        "instance_type": dig(m, "hardware", "instance_type", default=None),
        "region": dig(m, "hardware", "region", default=None),
        "lambda_instance_id": dig(m, "hardware", "lambda_instance_id", default=None),
        "consent_class": dig(m, "flags", "consent_class", default=None),
        "nonconformant": fs["nonconformant"],
        "nonconformant_reasons": fs["nonconformant_reasons"],
        "provenance_incomplete": fs["provenance_incomplete"],
        "provenance_incomplete_reasons": fs["provenance_incomplete_reasons"],
        "legacy_single_flag_manifest": fs["legacy_single_flag"],
        "exploratory": dig(m, "flags", "exploratory", default=False),
        "grading_degraded": dig(m, "flags", "grading_degraded", default=False),
        "checksums": run.get("checksums", "unknown"),
        "manifest_path": str(run["manifest_path"]),
        "run_dir": str(run["run_dir"]),
        "included": included,
    }


# ---------------------------------------------------------------- CLI


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors; this project reserves 2 for validation refusals."""

    def error(self, message):  # pragma: no cover - argparse plumbing
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="aggregate.py",
        description=(
            "Aggregate run bundles into the publication tables for 'The Harness Variable'. "
            "Consumes an explicit list of run manifests; directories are refused by design."
        ),
    )
    src = p.add_argument_group("inputs (at least one required)")
    src.add_argument("--manifests", nargs="+", metavar="PATH",
                     help="explicit run-manifest.json paths")
    src.add_argument("--manifest-list", nargs="+", metavar="FILE",
                     help="file(s) with one manifest path per line (# comments allowed)")
    src.add_argument("--results-root", nargs="+", metavar="DIR", default=[],
                     help="where unpacked run bundles live, if not next to the manifest")
    src.add_argument("--api-pricing", metavar="FILE",
                     help="api-pricing/v1 JSON for API-baseline models billed per token")
    src.add_argument("--cost-log", nargs="+", metavar="FILE", default=[],
                     help="CI cost-log.jsonl file(s); setup cost is reported separately")

    out = p.add_argument_group("outputs")
    out.add_argument("--out-dir", metavar="DIR", default="analysis/tables",
                     help="directory for summary.md / summary.json / *.csv (default: analysis/tables)")
    out.add_argument("--print", dest="print_fmt", choices=("md", "json", "none"), default="md",
                     help="what to write to stdout (default: md)")
    out.add_argument("--no-write", action="store_true", help="do not write any files")

    beh = p.add_argument_group("selection and validation")
    beh.add_argument("--allow-mixed", action="store_true",
                     help="aggregate across differing harness versions / prompt hashes and "
                          "annotate every output loudly")
    beh.add_argument("--strict", action="store_true",
                     help="treat comparability drift (adapter/inference/config) as fatal too")
    beh.add_argument("--include-partial", action="store_true",
                     help="include runs whose manifest status is not 'complete'")
    beh.add_argument("--include-exploratory", action="store_true",
                     help="include runs flagged exploratory (--limit/--instance debug runs)")
    beh.add_argument("--include-nonconformant", action="store_true",
                     help="include runs flagged nonconformant — a genuine harness deviation "
                          "(non-default budget, dirty repo, unresolved weight revision, prompt "
                          "drift). Runs flagged only provenance_incomplete are ALWAYS included; "
                          "their cost columns are annotated approximate instead")
    beh.add_argument("--include-truncated", action="store_true",
                     help="include runs whose instance list was truncated")
    beh.add_argument("--no-verify-checksums", action="store_true",
                     help="skip SHA256SUMS verification (faster, less safe)")
    beh.add_argument("--require-checksums", action="store_true",
                     help="fail when a run has no SHA256SUMS")
    beh.add_argument("--verify-refs", action="store_true",
                     help="also verify every patches/ and trajectories/ file against SHA256SUMS")
    beh.add_argument("--lenient", action="store_true",
                     help="downgrade per-record validation failures to warnings")
    beh.add_argument("--bootstrap-iters", type=int, default=10000, metavar="N",
                     help="bootstrap resamples for the resolve-rate CI (default: 10000)")
    beh.add_argument("--contamination-threshold", type=float, default=0.10, metavar="X",
                     help="Verified-gap fraction that raises the contamination flag (default: 0.10)")
    return p


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.manifests and not args.manifest_list:
        parser.error("one of --manifests or --manifest-list is required "
                     "(aggregate.py never globs an ambient results/ directory)")
    if args.bootstrap_iters < 0:
        parser.error("--bootstrap-iters must be >= 0")

    diag = Diagnostics()
    manifest_paths = read_manifest_paths(args, diag)
    if not manifest_paths and diag.errors:
        for e in diag.errors:
            print(f"error: {e['message']}", file=sys.stderr)
        return 3
    if not manifest_paths:
        print("error: no manifests given", file=sys.stderr)
        return 1

    results_roots = [Path(r) for r in args.results_root]
    api_pricing = load_api_pricing(Path(args.api_pricing) if args.api_pricing else None, diag)
    setup_costs = load_cost_logs([Path(c) for c in args.cost_log], diag)

    # ---- load
    loaded: list[dict] = []
    seen_run_ids: dict[str, Path] = {}
    io_failed = False
    for mp in manifest_paths:
        manifest = load_manifest(mp, diag)
        if manifest is None:
            io_failed = True
            continue
        run_id = manifest["run_id"]
        if run_id in seen_run_ids:
            diag.error(
                "duplicate_run_id",
                f"run_id {run_id} given twice ({seen_run_ids[run_id]} and {mp})",
            )
            continue
        seen_run_ids[run_id] = mp
        run_dir = locate_run_dir(mp, run_id, results_roots)
        if run_dir is None:
            diag.error(
                "results_not_found",
                f"{run_id}: no results.jsonl found next to {mp} or under any --results-root "
                f"{[str(r) for r in results_roots]}",
            )
            io_failed = True
            continue
        checksums = verify_run_checksums(run_dir, args, diag, run_id)
        records, ok = load_records(run_dir, manifest, args, diag)
        if not ok or checksums == "failed":
            continue
        if manifest.get("run_id") != run_dir.name and run_dir.name != run_id:
            diag.warn(
                "run_dir_name_mismatch",
                f"{run_id}: run directory is named {run_dir.name!r} (§2.3 expects the run id)",
            )
        loaded.append(
            {
                "run_id": run_id,
                "manifest": manifest,
                "manifest_path": mp,
                "run_dir": run_dir,
                "records": records,
                "checksums": checksums,
            }
        )

    if diag.errors:
        for e in diag.errors:
            print(f"error: {e['message']}", file=sys.stderr)
        return 3 if io_failed else 2

    included, excluded = select_runs(loaded, args, diag)
    for run in excluded:
        print(
            f"==> excluded {run['run_id']}: {'; '.join(run['exclusion_reasons'])}",
            file=sys.stderr,
        )
    if not included:
        print("error: no runs left after eligibility filtering (see --include-* flags)",
              file=sys.stderr)
        return 2

    # A cost-log line keyed by something that is not a harness run_id joins to nothing and
    # would otherwise vanish silently (the CI ledger used to be keyed by the GitHub run id).
    unmatched = sorted(set(setup_costs) - {run["run_id"] for run in loaded})
    if unmatched:
        diag.warn(
            "cost_log_run_id_unmatched",
            "cost log carries setup cost for {} run_id(s) that match no manifest given here "
            "({}{}) — if these are not harness run_ids of the form "
            "<model>__<suite>__<ts>__<hex>, the ledger is keyed wrong and no setup cost will "
            "ever be attributed".format(
                len(unmatched),
                ", ".join(unmatched[:3]),
                "…" if len(unmatched) > 3 else "",
            ),
        )

    # flags.provenance_incomplete does not exclude — say plainly what stayed in and why.
    provenance_rows = provenance_notes(included)
    if provenance_rows:
        print(
            f"==> {len(provenance_rows)} included run(s) carry flags.provenance_incomplete: the "
            "science is intact, the cost columns for their groups are APPROXIMATE (≈).",
            file=sys.stderr,
        )
        for row in provenance_rows:
            unresolved = ", ".join(row["unresolved"]) or "(no reasons recorded)"
            print(f"    {row['run_id']}: unresolved {unresolved}", file=sys.stderr)

    comparability = comparability_report(included, args, diag)
    if diag.errors:
        for e in diag.errors:
            print(f"error: {e['message']}", file=sys.stderr)
        print(
            "error: refusing to aggregate. Pass --allow-mixed only if you intend to publish a "
            "loudly-annotated mixed-harness table.",
            file=sys.stderr,
        )
        return 2

    # ---- group and compute
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in included:
        model = dig(run["manifest"], "model", "name", default="?")
        suite = dig(run["manifest"], "suite", "name", default="?")
        grouped.setdefault((model, suite), []).append(run)

    def suite_key(suite: str) -> int:
        return SUITE_ORDER.index(suite) if suite in SUITE_ORDER else len(SUITE_ORDER)

    groups = [
        compute_group(model, suite, runs, setup_costs, api_pricing, args, diag)
        for (model, suite), runs in sorted(grouped.items(), key=lambda kv: (kv[0][0], suite_key(kv[0][1])))
    ]

    cost_errors = [e for e in diag.errors if e["code"] in ("api_pricing_unavailable", "cost_unresolvable")]
    if cost_errors:
        for e in cost_errors:
            print(f"error: {e['message']}", file=sys.stderr)
        return 2

    report = {
        "schema": REPORT_SCHEMA,
        "aggregator_version": AGGREGATOR_VERSION,
        "generated_at": utcnow(),
        "options": {
            "allow_mixed": bool(args.allow_mixed),
            "strict": bool(args.strict),
            "include_partial": bool(args.include_partial),
            "include_exploratory": bool(args.include_exploratory),
            "include_nonconformant": bool(args.include_nonconformant),
            "include_truncated": bool(args.include_truncated),
            "provenance_incomplete_policy": "included; cost columns annotated approximate",
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "contamination_threshold": args.contamination_threshold,
            "checksum_verification": "skipped" if args.no_verify_checksums else "enabled",
            "verify_refs": bool(args.verify_refs),
        },
        "inputs": {
            "manifests": [str(p) for p in manifest_paths],
            "results_roots": [str(p) for p in results_roots],
            "api_pricing_file": args.api_pricing,
            "api_pricing_captured_at": api_pricing.get("captured_at") if api_pricing else None,
            "cost_logs": [str(c) for c in args.cost_log],
        },
        "comparability": comparability,
        "runs": [run_summary_row(r, api_pricing, True) for r in included],
        "excluded_runs": [
            dict(run_summary_row(r, api_pricing, False), exclusion_reasons=r["exclusion_reasons"])
            for r in excluded
        ],
        "provenance_incomplete_runs": provenance_rows,
        "by_model_suite": groups,
        "by_model": rollup_by_model(groups),
        "contamination": contamination_view(groups, args.contamination_threshold),
        "setup_costs_by_run": setup_costs,
        "diagnostics": {"warnings": diag.warnings, "errors": diag.errors},
    }

    markdown = render_markdown(report)

    if not args.no_write:
        out_dir = Path(args.out_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
            (out_dir / "summary.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            csv_by_model_suite(report, out_dir / "by_model_suite.csv")
            csv_failures(report, out_dir / "failures.csv")
            csv_contamination(report, out_dir / "contamination.csv")
            csv_runs(report, out_dir / "runs.csv")
        except OSError as exc:
            print(f"error: writing outputs to {args.out_dir}: {exc}", file=sys.stderr)
            return 3
        print(f"==> wrote {out_dir}/summary.md, summary.json, and 4 CSVs", file=sys.stderr)

    for w in diag.warnings:
        print(f"==> warning: {w['message']}", file=sys.stderr)
    if comparability["mixed"]:
        print(
            "==> MIXED-HARNESS AGGREGATE: a verdict-affecting knob differs across runs; outputs "
            "are annotated; do not publish as a like-for-like comparison",
            file=sys.stderr,
        )
    if any(g["cost_approximate"] for g in groups):
        print(
            "==> cost columns marked ≈ are approximate (provenance_incomplete runs included by "
            "design); resolve rates and token counts are exact",
            file=sys.stderr,
        )

    if args.print_fmt == "md":
        sys.stdout.write(markdown)
    elif args.print_fmt == "json":
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("==> interrupted", file=sys.stderr)
        sys.exit(130)
