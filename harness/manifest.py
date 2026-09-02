#!/usr/bin/env python3
# harness/manifest.py — provenance collector and run-directory bookkeeper for harness/run.sh.
#
#   manifest.py build        ...  resolve every provenance field, write run-manifest.json
#   manifest.py finalize     ...  rewrite status/timing/flags, write run-status.json
#   manifest.py prompt-check ...  assert the adapter renders the manifest's template id
#   manifest.py grading-preflight  refuse (exit 3) when this host cannot grade the suite
#   manifest.py scan         ...  attempt + failure histogram of results.jsonl (KEY=VALUE)
#   manifest.py missing      ...  instance ids with no record for a pass (for --resume)
#   manifest.py prune-retryable    drop a pass's INFRA_HOST records so --resume can replace them
#   manifest.py checksums    ...  write SHA256SUMS over a run directory
#   manifest.py get          ...  print one dotted field of a JSON file
#   manifest.py dir-digest   ...  print the §2.4 digest of a directory (test hook)
#
# Implements docs/CONTRACTS.md §2 (run-manifest/v1) and §2.4 (directory digest).
# Python 3.11 stdlib only. Never touches the network unless HARNESS_ALLOW_NETWORK=1.
#
# Exit codes match run.sh: 0 ok, 2 config error, 3 unresolved REQUIRED provenance.
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import ipaddress
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

MANIFEST_SCHEMA = "run-manifest/v1"
RESULT_SCHEMA = "raw-result/v1"
STUDY_SEED = 20260830

# suite -> (adapter module path, default seed file, consent class)   [CONTRACTS.md §5.4]
SUITES = {
    "swebench-verified": ("harness/adapters/swebench_verified.py", "suites/verified-100.json", "public"),
    "swebench-pro": ("harness/adapters/swebench_pro.py", "suites/pro-50.json", "public"),
    "agenttask": ("harness/adapters/agenttask.py", "suites/agenttask/seed.json", "restricted"),
}

# CONTRACTS.md §2.4 — skipped everywhere a directory digest is computed.
SKIP_DIRS = {".git", ".cache", "__pycache__"}
SKIP_NAMES = {".DS_Store"}

ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Contract defaults for the held-constant knobs; harness/agent_config.json is the
# source of truth and these only fill gaps in it.  [CONTRACTS.md §2.1 "inference"]
INFERENCE_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "seed": STUDY_SEED,
    "max_tokens": 8192,
    "stop": [],
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_iters": 40,
    "max_attempt_tokens": 600000,
    "task_timeout_s": 1800,
    "concurrency": 4,
    "retry_policy": {
        "max_retries": 3,
        "retry_on": ["http_5xx", "connection_error", "empty_response", "malformed_tool_call"],
        "backoff": "exponential",
        "base_delay_s": 2,
        "max_delay_s": 30,
        "jitter": False,
        "retries_count_against_iteration_budget": False,
    },
}

# CONTRACTS.md §0.1: --max-model-len is a HELD CONSTANT of the study, exactly like the
# iteration budget. models.d/<model>.env is the file a future contributor edits per model,
# so the value it carries is checked against this constant on every build.
STUDY_MAX_MODEL_LEN = 262144

UNRESOLVED = "unresolved"

# Serving flags that CONTRACTS.md §0.1 holds constant across models. modelctl places
# $EXTRA_ARGS *before* these on the vllm command line so that under argparse last-wins the
# study values always win; a models.d/<model>.env that carries one of them anyway is a
# nonconformant run, not a silently-overridden one.
HELD_CONSTANT_SERVING_FLAGS = ("--max-model-len", "--served-model-name")

# Records carrying this error code were never scored: the host itself went away (SIGTERM,
# reaper, disk full). A --resume replaces them rather than counting them as done (§3.1).
RETRYABLE_ERROR_CODES = ("INFRA_HOST",)

# Per-file hashing threads for dir_digest. hashlib releases the GIL, so sha256 over a
# multi-hundred-GB weights tree scales with cores; the digest itself is order-independent
# of how the hashes were produced (§2.4 sorts the stream afterwards).
DIGEST_WORKERS = min(32, max(4, (os.cpu_count() or 4) * 2))


# --------------------------------------------------------------------------- basics

def warn(msg: str) -> None:
    sys.stderr.write("==> manifest: %s\n" % msg)


def die(msg: str, code: int = 2) -> "NoReturn":  # noqa: F821
    sys.stderr.write("error: manifest: %s\n" % msg)
    raise SystemExit(code)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def id_set_sha256(ids) -> str:
    """CONTRACTS.md: sha256("\\n".join(sorted(ids)) + "\\n") — order independent."""
    return sha256_text("\n".join(sorted(ids)) + "\n")


def write_json(path, obj) -> None:
    """Canonical on-disk JSON: UTF-8, indent=2, sort_keys, one trailing newline."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sh(args, cwd=None, timeout=60):
    """Run a command, return stripped stdout or None. Never raises."""
    try:
        out = subprocess.run(
            args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.decode("utf-8", "replace").strip()
    return text or None


# ------------------------------------------------------------------ directory digest

def walk_files(root: Path):
    """Yield (relpath, abspath) for CONTRACTS.md §2.4, plus the symlinks skipped."""
    files = []
    links = []
    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=False):
        keep = []
        for d in sorted(dirnames):
            full = os.path.join(dirpath, d)
            if d in SKIP_DIRS:
                continue
            if os.path.islink(full):
                links.append(os.path.relpath(full, str(root)))
                continue
            keep.append(d)
        dirnames[:] = keep
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if name in SKIP_NAMES or name.endswith(".pyc"):
                continue
            if os.path.islink(full):
                links.append(os.path.relpath(full, str(root)))
                continue
            rel = os.path.relpath(full, str(root)).replace(os.sep, "/")
            files.append((rel, full))
    return files, links


class SymlinkError(ValueError):
    """A weights tree contained a symlink; §2.4 makes that fatal for weights."""


def dir_digest(root: Path, symlinks_fatal: bool = False, workers=None):
    """CONTRACTS.md §2.4. Returns (hexdigest, file_count, total_bytes).

    ``workers`` > 1 hashes files on a thread pool; ``workers == 1`` is the plain serial
    reference. Both produce the identical byte stream: the per-file hashes are a pure
    function of the bytes, and the ``(rel, h)`` pairs are sorted afterwards exactly as
    §2.4 step 3 requires, so the result cannot depend on completion order.
    """
    files, links = walk_files(root)
    if links and symlinks_fatal:
        raise SymlinkError("%s contains symlinks (%s ...) — weights must be real files"
                           % (root, links[0]))
    if links:
        warn("%s: skipped %d symlink(s) per §2.4" % (root, len(links)))
    if workers is None:
        workers = DIGEST_WORKERS
    paths = [full for _rel, full in files]
    if workers > 1 and len(paths) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(paths))) as ex:
            hashes = list(ex.map(sha256_file, paths))
    else:
        hashes = [sha256_file(full) for full in paths]
    pairs = [(rel, h) for (rel, _full), h in zip(files, hashes)]
    total = sum(os.path.getsize(full) for full in paths)
    pairs.sort(key=lambda p: p[0].encode("utf-8"))
    stream = "".join("%s  %s\n" % (h, rel) for rel, h in pairs)
    return sha256_text(stream), len(pairs), total


def dir_fingerprint(root: Path) -> str:
    """Cheap (path, size, mtime) fingerprint — the cache key for a weight digest."""
    files, _ = walk_files(root)
    meta = []
    for rel, full in files:
        st = os.stat(full)
        meta.append([rel, st.st_size, st.st_mtime_ns])
    meta.sort()
    return sha256_text(json.dumps(meta, sort_keys=True, separators=(",", ":")))


def weight_digest_cache_dir(root: Path) -> Path:
    """Where the digest cache for ``root`` lives.

    Weights sit on the persistent filesystem while ``$HOME`` is the ephemeral root disk of
    a freshly dispatched CI instance, so a cache under ``~`` is cold on every dispatch and
    a multi-hundred-GB rehash lands on the critical path after vLLM is already billing.
    The cache therefore lives *next to* the weights — ``<weights_dir>/../.harness/
    weight-digest/`` — on the same filesystem but OUTSIDE the digested tree, so it can
    never perturb the digest it describes (§2.4 does not skip ``.harness/``, so putting it
    inside the tree would change the digest on the second run). ``~/.harness`` is the
    fallback only when that location is not writable.
    """
    candidates = [root.parent / ".harness" / "weight-digest",
                  Path.home() / ".harness" / "weight-digest"]
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            if os.access(str(cand), os.W_OK):
                return cand
        except OSError:
            continue
    return candidates[-1]


def cached_dir_digest(root: Path, cache_key: str):
    """Digest a very large tree once. Cache lives outside the tree so it cannot
    perturb the digest it describes (see weight_digest_cache_dir)."""
    cache = weight_digest_cache_dir(root) / (cache_key + ".json")
    fp = dir_fingerprint(root)
    try:
        blob = read_json(cache)
        if blob.get("fingerprint") == fp and blob.get("digest"):
            warn("weight digest: cache hit (%s)" % cache)
            return blob["digest"], int(blob["file_count"]), int(blob["bytes"])
    except (OSError, ValueError, KeyError):
        pass
    warn("weight digest: hashing %s (this is slow for large models)" % root)
    digest, count, total = dir_digest(root, symlinks_fatal=True)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_json(cache, {"fingerprint": fp, "digest": digest,
                           "file_count": count, "bytes": total, "root": str(root)})
    except OSError as exc:
        warn("weight digest: could not cache (%s)" % exc)
    return digest, count, total


# ----------------------------------------------------------------------- provenance

class Resolver:
    """Collects values, remembering which REQUIRED ones could not be resolved and how a
    degraded resolution has to be classified.

    CONTRACTS.md §2.2 keeps these two apart, because conflating them either throws away
    good science or publishes bad science:

      * ``nonconformant``          — a genuine harness deviation that breaks comparability:
        dirty repo, non-default ``--max-iters`` or ``MAX_MODEL_LEN``, unresolved weight
        revision/digest, prompt-template drift.  ``analysis/aggregate.py`` excludes these
        runs from headline numbers by default.
      * ``provenance_incomplete``  — the science is intact but cost/provenance attribution
        is imprecise: missing ``LAMBDA_INSTANCE_ID``/``LAMBDA_REGION``, fallback pricing,
        an unresolved requirements-lock hash.  Included by default; the cost columns are
        annotated as approximate and the unresolved fields are named.

    Both are write-once booleans plus a de-duplicated list of reasons.
    """

    def __init__(self, strict: bool):
        self.strict = strict
        self.unresolved = []
        self.nonconformant_reasons = []
        self.provenance_reasons = []

    @property
    def reasons(self):
        """Every reason, nonconformant first — what `notes` records."""
        return list(self.nonconformant_reasons) + list(self.provenance_reasons)

    def nonconformant(self, reason: str) -> None:
        if reason not in self.nonconformant_reasons:
            self.nonconformant_reasons.append(reason)
            warn("nonconformant: " + reason)

    def provenance(self, reason: str) -> None:
        # A reason already recorded as a comparability break is not downgraded.
        if reason in self.nonconformant_reasons:
            return
        if reason not in self.provenance_reasons:
            self.provenance_reasons.append(reason)
            warn("provenance incomplete: " + reason)

    def required(self, name: str, value, sentinel=UNRESOLVED, kind: str = "nonconformant"):
        if value in (None, "", []):
            if kind == "provenance":
                # A provenance gap degrades cost attribution, not comparability: it must NOT
                # feed self.unresolved, which cmd_build turns into a hard exit 3. That would
                # defeat the whole point of splitting the flag.
                self.provenance("%s unresolved" % name)
            else:
                self.unresolved.append(name)
                self.nonconformant("%s unresolved" % name)
            return sentinel
        return value


def git_facts(repo: Path, res: Resolver):
    sha = sh(["git", "-C", str(repo), "rev-parse", "HEAD"])
    describe = sh(["git", "-C", str(repo), "describe", "--tags", "--always", "--dirty"])
    porcelain = sh(["git", "-C", str(repo), "status", "--porcelain"])
    inside = sh(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"])
    if inside != "true":
        res.nonconformant("not a git work tree — repo provenance unavailable")
        return res.required("harness.repo_git_sha", None), None, False
    dirty = bool(porcelain)
    if dirty:
        res.nonconformant("repo has uncommitted changes (git status --porcelain non-empty)")
    return res.required("harness.repo_git_sha", sha), describe, dirty


def extra_args_overriding_held_constants(extra_args: str):
    """The held-constant serving flags that ``EXTRA_ARGS`` tries to set, in order."""
    try:
        tokens = shlex.split(extra_args or "")
    except ValueError:
        tokens = (extra_args or "").split()
    hits = []
    for tok in tokens:
        name = tok.split("=", 1)[0]
        name = name.replace("_", "-")  # argparse accepts --max_model_len too
        if name in HELD_CONSTANT_SERVING_FLAGS and name not in hits:
            hits.append(name)
    return hits


def endpoint_is_loopback(endpoint: str) -> bool:
    """True when the --endpoint host is localhost / 127.0.0.1 / ::1 (self-hosted vLLM)."""
    raw = (endpoint or "").strip()
    if "://" not in raw:
        raw = "http://" + raw
    try:
        host = urlsplit(raw).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def model_env(key: str, default: str = "") -> str:
    """Values run.sh obtained by sourcing models.d/<model>.env the way modelctl does."""
    return os.environ.get("HARNESS_MODELENV_" + key, default)


def resolve_weight_revision(weights_dir: Path, hf_repo: str, res: Resolver):
    pinned = model_env("HF_REVISION").strip()
    if pinned:
        return pinned, "pinned_env"

    hashes = set()
    cache_root = weights_dir / ".cache" / "huggingface" / "download"
    if cache_root.is_dir():
        for dirpath, _dirnames, filenames in os.walk(str(cache_root)):
            for name in filenames:
                if not name.endswith(".metadata"):
                    continue
                try:
                    with open(os.path.join(dirpath, name), "r", encoding="utf-8") as fh:
                        first = fh.readline().strip()
                except OSError:
                    continue
                if re.fullmatch(r"[0-9a-f]{40}", first or ""):
                    hashes.add(first)
    if len(hashes) == 1:
        return hashes.pop(), "hf_cache_metadata"
    if len(hashes) > 1:
        res.nonconformant("hf download metadata sidecars disagree on the commit hash")

    if os.environ.get("HARNESS_ALLOW_NETWORK") == "1" and hf_repo:
        try:
            from huggingface_hub import HfApi  # type: ignore

            sha = HfApi().model_info(hf_repo).sha
            if sha:
                return sha, "hf_api"
        except Exception as exc:  # noqa: BLE001 - any failure just drops a rung
            warn("hf_api revision lookup failed: %s" % exc)

    res.nonconformant("weight_revision unresolved — pin HF_REVISION in models.d/%s.env"
                      % model_env("NAME", "<model>"))
    return UNRESOLVED, "unresolved"


def dist_version(name: str):
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def vllm_dist_digest(version: str):
    """sha256 of the installed distribution's RECORD — pins the artifact, not the string."""
    try:
        import importlib.metadata as md

        dist = md.distribution("vllm")
    except Exception:  # noqa: BLE001
        return None
    base = getattr(dist, "_path", None)
    if base is not None:
        record = Path(base) / "RECORD"
        if record.is_file():
            return "sha256:" + sha256_file(record)
    try:
        text = dist.read_text("RECORD")
    except Exception:  # noqa: BLE001
        text = None
    if not text:
        return None
    return "sha256:" + sha256_text(text)


def nvidia_smi_facts():
    out = {"driver": None, "cuda": None, "gpu_model": None, "gpu_count": None, "gpu_mib": None}
    drv = sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if drv:
        out["driver"] = drv.splitlines()[0].strip()
    gpus = sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if gpus:
        lines = [ln for ln in gpus.splitlines() if ln.strip()]
        out["gpu_count"] = len(lines)
        first = lines[0].split(",")
        out["gpu_model"] = first[0].strip()
        if len(first) > 1:
            mib = re.search(r"(\d+)", first[1])
            if mib:
                out["gpu_mib"] = int(mib.group(1))
    header = sh(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", header)
        if m:
            out["cuda"] = m.group(1)
    if out["cuda"] is None:
        tv = dist_version("torch")
        if tv and "+cu" in tv:
            digits = tv.split("+cu")[-1]
            if digits.isdigit() and len(digits) >= 3:
                out["cuda"] = "%s.%s" % (digits[:-1], digits[-1])
    return out


def resolve_hardware(res: Resolver, node_count: int):
    """Ladder: CI env -> ~/.harness/instance.json -> local probe -> null.
    Per the CONTRACTS.md §2.2 CI note this ladder degrades instead of exiting 3."""
    fromfile = {}
    inst_file = Path.home() / ".harness" / "instance.json"
    if inst_file.is_file():
        try:
            fromfile = read_json(inst_file)
        except ValueError:
            warn("%s is not valid JSON — ignored" % inst_file)
    used = set()

    def pick(env_name, file_key):
        val = os.environ.get(env_name, "").strip()
        if val:
            used.add("env")
            return val
        val = str(fromfile.get(file_key, "") or "").strip()
        if val:
            used.add("file")
            return val
        return None

    instance_type = pick("LAMBDA_INSTANCE_TYPE", "instance_type") or model_env("INSTANCE_TYPE") or None
    instance_id = pick("LAMBDA_INSTANCE_ID", "instance_id")
    region = pick("LAMBDA_REGION", "region")
    instance_name = pick("INSTANCE_NAME", "instance_name")

    gpu = nvidia_smi_facts()
    if gpu["gpu_model"]:
        used.add("probe")
    hostname = sh(["hostname", "-f"]) or sh(["hostname"]) or platform.node() or None

    # These are cost/provenance attribution, not comparability: the model, the harness and
    # the grading are unchanged by a missing instance id (CONTRACTS.md §2.2).
    if not instance_type:
        res.provenance("hardware.instance_type unresolved — the price ladder has no key")
    if not instance_id:
        res.provenance("hardware.lambda_instance_id unresolved — cost cannot be "
                       "reconciled against Lambda billing")
    if not region:
        res.provenance("hardware.region unresolved")

    if "env" in used:
        provenance = "env"
    elif "file" in used:
        provenance = "file"
    elif "probe" in used:
        provenance = "probe"
    else:
        provenance = "unknown"

    return {
        "instance_type": instance_type,
        "region": region,
        "lambda_instance_id": instance_id,
        "instance_name": instance_name,
        "hostname": hostname,
        "gpu_model": gpu["gpu_model"],
        "gpu_count": gpu["gpu_count"],
        "gpu_memory_total_mib": gpu["gpu_mib"],
        "node_count": node_count,
        "provenance": provenance,
    }, gpu


def resolve_price(repo: Path, instance_type, node_count: int, res: Resolver):
    """Ladder: $HARNESS_PRICE_SNAPSHOT -> ./lambdactl types -> pricing/fallback-prices.json."""
    empty = {
        "source": UNRESOLVED, "captured_at": None, "instance_type": instance_type,
        "price_cents_per_hour": None, "currency": "USD", "regions_with_capacity": [],
        "node_count": node_count, "effective_cents_per_hour": None,
    }
    if not instance_type:
        res.provenance("price unresolved — no instance type")
        return empty

    def finish(source, captured_at, cents, regions):
        return {
            "source": source,
            "captured_at": captured_at,
            "instance_type": instance_type,
            "price_cents_per_hour": int(cents),
            "currency": "USD",
            "regions_with_capacity": list(regions),
            "node_count": node_count,
            "effective_cents_per_hour": int(cents) * node_count,
        }

    def from_snapshot(path, source):
        try:
            blob = read_json(path)
        except (OSError, ValueError) as exc:
            warn("price snapshot %s unreadable (%s)" % (path, exc))
            return None
        entry = (blob.get("prices") or {}).get(instance_type)
        if not entry or entry.get("price_cents_per_hour") is None:
            warn("price snapshot %s has no entry for %s" % (path, instance_type))
            return None
        return finish(source, blob.get("captured_at"), entry["price_cents_per_hour"],
                      entry.get("regions_with_capacity") or [])

    snap = os.environ.get("HARNESS_PRICE_SNAPSHOT", "").strip()
    if snap:
        got = from_snapshot(snap, "snapshot-file")
        if got:
            return got

    lambdactl = repo / "lambdactl"
    if lambdactl.is_file() and os.access(str(lambdactl), os.X_OK) and os.environ.get("LAMBDA_API_KEY"):
        text = sh([str(lambdactl), "types"], cwd=str(repo), timeout=60)
        if text:
            for line in text.splitlines():
                m = re.match(r"^(\S+)\s+\$\s*([0-9]+(?:\.[0-9]+)?)/hr\s*(.*)$", line)
                if m and m.group(1) == instance_type:
                    # `lambdactl types` prints an em dash when no region has capacity;
                    # that means [], not ["—"].
                    tail = m.group(3).strip()
                    regions = [] if tail in ("", "-", "\u2014", "none") else \
                        [r.strip() for r in tail.split(",")
                         if r.strip() and r.strip() not in ("-", "\u2014")]
                    cents = int(round(float(m.group(2)) * 100))
                    return finish("lambdactl-types", utcnow(), cents, regions)
        warn("./lambdactl types did not report %s" % instance_type)

    fallback = repo / "pricing" / "fallback-prices.json"
    if fallback.is_file():
        got = from_snapshot(fallback, "static-fallback")
        if got:
            res.provenance("price came from pricing/fallback-prices.json, not a live "
                           "snapshot — cost figures are approximate")
            return got

    res.provenance("price unresolved for %s — cost cannot be computed" % instance_type)
    return empty


def resolve_adapter(repo: Path, suite: str, res: Resolver):
    rel_path, _seed, _consent = SUITES[suite]
    path = repo / rel_path
    version = None
    # §5.4: prefer the registry so the recorded path is the real module file.
    sys.path.insert(0, str(repo))
    try:
        import importlib

        registry = importlib.import_module("harness.adapters")
        mod = registry.get(suite)
        modfile = getattr(mod, "__file__", None)
        if modfile:
            path = Path(modfile).resolve()
            try:
                rel_path = str(path.relative_to(repo))
            except ValueError:
                rel_path = str(path)
        version = getattr(mod, "ADAPTER_VERSION", None)
    except Exception as exc:  # noqa: BLE001 - registry not importable yet, use the table
        warn("adapter registry not importable (%s) — falling back to the §5.4 table" % exc)
    finally:
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)

    if not path.is_file():
        die("adapter for suite '%s' not found at %s" % (suite, path), 2)
    if version is None:
        text = path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"""^ADAPTER_VERSION\s*(?::[^=]+)?=\s*["']([^"']+)["']""", text, re.M)
        version = m.group(1) if m else None
    return rel_path, sha256_file(path), res.required("harness.adapter_version", version)


def adapters_dir_digest(repo: Path, res: Resolver):
    """Directory digest (CONTRACTS.md §2.4) of `harness/adapters/`.

    `harness.adapter_sha256` covers only the suite's own module, and for both SWE-bench
    suites that module is a ~70-line delegating shim: task construction, prompt rendering,
    eval invocation and verdict mapping all live in `harness/adapters/_swebench.py`, which
    no other field hashes. Without this digest an edit to the grading logic between two
    model runs would be invisible in the manifests, and the two runs would look comparable
    when they are not.
    """
    root = repo / "harness" / "adapters"
    if not root.is_dir():
        res.nonconformant("harness/adapters/ is missing — the grading code cannot be digested")
        return None
    try:
        digest, _count, _bytes = dir_digest(root)
    except (OSError, SymlinkError) as exc:
        res.nonconformant("harness/adapters/ digest failed (%s) — the grading code is "
                          "not pinned by this manifest" % exc)
        return None
    return digest


def load_adapter_module(repo: Path, suite: str):
    """Import a suite adapter through the §5.4 registry, falling back to the §5.4 table.

    Raises on failure — callers decide whether that is fatal.
    """
    import importlib

    added = False
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
        added = True
    try:
        try:
            return importlib.import_module("harness.adapters").get(suite)
        except (ImportError, AttributeError, KeyError):
            name = Path(SUITES[suite][0]).stem
            return importlib.import_module("harness.adapters." + name)
    finally:
        if added and sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)


def resolve_environment_digest(repo: Path, suite: str, res: Resolver):
    """`harness.environment_digest` — the identity of the GRADING environment.

    CONTRACTS.md §5.3 makes `grade()` deterministic only given
    (task, patch, environment_digest()). Every adapter computes this value; recording it
    here is what lets two runs be compared at all, and what lets a verdict be reproduced.
    """
    try:
        mod = load_adapter_module(repo, suite)
        func = getattr(mod, "environment_digest", None)
        if not callable(func):
            raise AttributeError("the %s adapter exposes no environment_digest()" % suite)
        digest = func()
    except Exception as exc:  # noqa: BLE001 - never crash the manifest write
        res.nonconformant("harness.environment_digest unresolved (%s: %s) — the grading "
                          "environment cannot be compared across runs"
                          % (type(exc).__name__, exc))
        return None
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        res.nonconformant("the %s adapter's environment_digest() returned %r, not "
                          "'sha256:<hex>'" % (suite, digest))
        return None
    return digest


def load_seed_file(path: Path, suite: str):
    try:
        blob = read_json(path)
    except OSError as exc:
        die("seed file %s: %s" % (path, exc), 2)
    except ValueError as exc:
        die("seed file %s is not valid JSON: %s" % (path, exc), 2)
    if blob.get("schema") != "suite-seed/v1":
        die("seed file %s has schema %r, expected suite-seed/v1" % (path, blob.get("schema")), 2)
    if blob.get("suite") != suite:
        die("seed file %s is for suite %r, not %r" % (path, blob.get("suite"), suite), 2)
    ids = blob.get("instance_ids")
    if not isinstance(ids, list) or not ids:
        die("seed file %s has no instance_ids" % path, 2)
    bad = [i for i in ids if not isinstance(i, str) or not ID_RE.match(i)]
    if bad:
        die("seed file %s: instance id %r does not match ^[A-Za-z0-9._-]+$" % (path, bad[0]), 2)
    if len(set(ids)) != len(ids):
        die("seed file %s contains duplicate instance ids" % path, 2)
    if blob.get("count") != len(ids):
        die("seed file %s: count=%r but %d instance_ids" % (path, blob.get("count"), len(ids)), 2)
    # instance_ids_sha256 is the freeze seal (§6.1). It is REQUIRED for a real seed file:
    # an absent hash is not "unfrozen", it is the signature of an edited id list whose
    # stale hash was deleted — which would otherwise be re-sealed by the build below.
    recorded = blob.get("instance_ids_sha256")
    placeholder = bool(blob.get("placeholder"))
    if not recorded:
        if placeholder:
            warn("seed file %s is a placeholder without instance_ids_sha256" % path)
        else:
            die("seed file %s has no instance_ids_sha256 — a frozen seed file MUST carry "
                "the seal of its id set (§6.1); regenerate it with suites/select.py or "
                "suites/generate_seeds.py rather than editing instance_ids" % path, 2)
    elif recorded != id_set_sha256(ids):
        die("seed file %s: instance_ids_sha256 mismatch — the id set has been edited "
            "since it was frozen" % path, 2)
    sel = blob.get("selection") or {}
    if sel.get("seed") is None or not sel.get("method"):
        die("seed file %s: selection.seed and selection.method are required" % path, 2)
    return blob, ids


def load_partitions(path: Path):
    try:
        blob = read_json(path)
    except OSError as exc:
        die("partitions file %s: %s" % (path, exc), 2)
    except ValueError as exc:
        die("partitions file %s is not valid JSON: %s" % (path, exc), 2)
    if blob.get("schema") != "partitions/v1":
        die("partitions file %s has schema %r, expected partitions/v1" % (path, blob.get("schema")), 2)
    if not isinstance(blob.get("partitions"), dict):
        die("partitions file %s has no partitions object" % path, 2)
    if blob.get("placeholder"):
        die("partitions file %s is a PLACEHOLDER — it was derived from placeholder seed "
            "files and is not a freeze (§6.2). Regenerate it with suites/generate_partitions.py "
            "together with the real seed files before any run" % path, 2)
    for name, block in blob["partitions"].items():
        if not isinstance(block, dict) or not isinstance(block.get("ids"), list):
            die("partitions file %s: partition %r is not a {count, ids} block" % (path, name), 2)
    return blob


def partitioned_ids(blob) -> set:
    """Every fully-qualified ``suite::instance_id`` that resolves to some partition."""
    out = set()
    for block in blob["partitions"].values():
        out.update(i for i in block["ids"] if isinstance(i, str))
    return out


def load_agent_config(repo: Path):
    path = repo / "harness" / "agent_config.json"
    if not path.is_file():
        die("harness/agent_config.json is missing — it is the source of truth for the "
            "held-constant inference knobs (CONTRACTS.md §2.2)", 2)
    try:
        blob = read_json(path)
    except ValueError as exc:
        die("harness/agent_config.json is not valid JSON: %s" % exc, 2)
    cfg = blob.get("inference") if isinstance(blob.get("inference"), dict) else blob
    merged = dict(INFERENCE_DEFAULTS)
    for key in INFERENCE_DEFAULTS:
        if key in cfg:
            merged[key] = cfg[key]
        else:
            warn("agent_config.json does not set %r — using the contract default %r"
                 % (key, INFERENCE_DEFAULTS[key]))
    return merged, sha256_file(path)


def read_vllm_argv(run_dir: Path):
    path = run_dir / "env" / "vllm-args.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text or text.startswith("unavailable"):
        return None
    return text.splitlines()[0].strip()


def read_quantization(weights_dir: Path):
    cfg = weights_dir / "config.json"
    if not cfg.is_file():
        return "none"
    try:
        blob = read_json(cfg)
    except ValueError:
        return "none"
    qc = blob.get("quantization_config") or {}
    return qc.get("quant_method") or "none"


# ---------------------------------------------------------------------------- build

def cmd_build(args) -> int:
    repo = Path(args.repo).resolve()
    run_dir = Path(args.run_dir).resolve()
    strict = args.mode == "exec"
    res = Resolver(strict)

    created_at = args.created_at or utcnow()

    # ---- harness -----------------------------------------------------------
    version_file = repo / "harness" / "VERSION"
    if not version_file.is_file():
        die("harness/VERSION is missing", 2)
    harness_version = version_file.read_text(encoding="utf-8").strip()

    git_sha, git_describe, git_dirty = git_facts(repo, res)

    prompts_dir = repo / "harness" / "prompts"
    template_file = prompts_dir / "TEMPLATE_ID"
    if not prompts_dir.is_dir() or not template_file.is_file():
        die("harness/prompts/ or harness/prompts/TEMPLATE_ID is missing", 2)
    template_id = template_file.read_text(encoding="utf-8").strip()
    if not template_id:
        die("harness/prompts/TEMPLATE_ID is empty", 2)
    prompt_dir_sha, _n, _b = dir_digest(prompts_dir)

    inference_cfg, agent_config_sha = load_agent_config(repo)
    adapter_rel, adapter_sha, adapter_version = resolve_adapter(repo, args.suite, res)
    adapters_dir_sha = adapters_dir_digest(repo, res)
    env_digest = resolve_environment_digest(repo, args.suite, res)

    invocation = []
    if args.invocation_file:
        raw = Path(args.invocation_file).read_bytes()
        invocation = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]

    # ---- suite -------------------------------------------------------------
    seed_path = Path(args.seed_file).resolve()
    seed_blob, all_ids = load_seed_file(seed_path, args.suite)

    ids = list(all_ids)
    if args.instance:
        wanted = list(dict.fromkeys(args.instance))
        unknown = [i for i in wanted if i not in set(all_ids)]
        if unknown:
            die("--instance %s is not in %s" % (unknown[0], seed_path), 2)
        ids = [i for i in all_ids if i in set(wanted)]
    if args.limit is not None:
        if args.limit < 1:
            die("--limit must be >= 1", 2)
        ids = ids[: args.limit]
    truncated = len(ids) != len(all_ids)

    partitions_path = Path(args.partitions).resolve()
    partitions_blob = load_partitions(partitions_path)
    # §6.2: the partitions file is frozen over exactly the seed sets. A seed set none of
    # whose ids is partitioned was never part of that freeze — every record would come out
    # partition:"unpartitioned" and the leakage guard would have nothing to guard.
    qualified = set("%s::%s" % (args.suite, i) for i in all_ids)
    if not qualified & partitioned_ids(partitions_blob):
        res.nonconformant("no seed id is partitioned — partitions.json was not frozen for "
                          "this seed set")

    def repo_rel(path: Path) -> str:
        try:
            return str(path.relative_to(repo))
        except ValueError:
            return str(path)

    # ---- model -------------------------------------------------------------
    hf_repo = res.required("model.hf_repo", model_env("HF_REPO"))
    weights_dir = Path(args.weights_dir).resolve()
    model_env_file = repo / "models.d" / (args.model + ".env")
    if not model_env_file.is_file():
        die("unknown model '%s' — no models.d/%s.env" % (args.model, args.model), 2)

    weight_digest, weight_files, weight_bytes = None, 0, 0
    if not weights_dir.is_dir():
        warn("weights directory %s does not exist — run: ./modelctl download %s"
             % (weights_dir, args.model))
    elif os.environ.get("HARNESS_SKIP_WEIGHT_DIGEST") == "1":
        # Documented escape hatch (run.sh header, CONTRACTS.md §2.2). Skipping the digest
        # costs comparability, but it MUST NOT turn into an exit-3 preflight failure — so
        # the sentinel is written here instead of falling through to res.required() below.
        weight_digest = UNRESOLVED
        res.nonconformant("weight digest skipped (HARNESS_SKIP_WEIGHT_DIGEST=1) — "
                          "model.weight_digest is the '%s' sentinel" % UNRESOLVED)
    else:
        try:
            digest, weight_files, weight_bytes = cached_dir_digest(weights_dir, args.model)
            weight_digest = "sha256:" + digest
        except SymlinkError as exc:
            warn(str(exc))
        except OSError as exc:
            warn("weight digest failed: %s" % exc)
    # REQUIRED: unresolved here means exit 3 in exec mode, but only after the manifest
    # has been written (CONTRACTS.md §1.3).
    if weight_digest != UNRESOLVED:
        weight_digest = res.required("model.weight_digest", weight_digest)

    weight_revision, weight_revision_source = resolve_weight_revision(weights_dir, hf_repo, res)

    # ---- runtime -----------------------------------------------------------
    vllm_version = dist_version("vllm")
    if not vllm_version:
        # The server may legitimately live on another host: this is a recording gap, not a
        # deviation from the held-constant harness.
        res.provenance("vllm is not importable here — runtime.vllm_version unresolved")
    docker_image = model_env("VLLM_DOCKER_IMAGE").strip() or None
    docker_digest = None
    if docker_image:
        docker_digest = sh(["docker", "inspect", "--format", "{{index .RepoDigests 0}}", docker_image])
        if not docker_digest:
            res.required("runtime.vllm_docker_image_digest", None, None, kind="provenance")

    pip_freeze = run_dir / "env" / "pip-freeze.txt"
    lock = repo / "harness" / "requirements.lock"
    lock_sha = sha256_file(lock) if lock.is_file() else None
    if lock_sha is None:
        res.provenance("harness/requirements.lock is missing — "
                       "runtime.requirements_lock_sha256 is null")
    multinode = model_env("MULTINODE", "0") == "1"
    node_count = 2 if multinode else 1

    def as_int(name, default):
        raw = model_env(name, "").strip()
        try:
            return int(raw)
        except ValueError:
            return default

    # CONTRACTS.md §0.1 holds --max-model-len constant across models, exactly like the
    # iteration budget checked further down. models.d/<model>.env is where a future
    # contributor would quietly change it per model, so it is checked here.
    raw_max_model_len = model_env("MAX_MODEL_LEN", "").strip()
    if raw_max_model_len and not raw_max_model_len.isdigit():
        res.nonconformant("MAX_MODEL_LEN=%r in models.d/%s.env is not an integer — "
                          "modelctl would launch with the default %d instead"
                          % (raw_max_model_len, args.model, STUDY_MAX_MODEL_LEN))
    max_model_len = as_int("MAX_MODEL_LEN", STUDY_MAX_MODEL_LEN)
    if max_model_len != STUDY_MAX_MODEL_LEN:
        res.nonconformant("MAX_MODEL_LEN=%d in models.d/%s.env is not the held-constant "
                          "context window %d (CONTRACTS.md §0.1)"
                          % (max_model_len, args.model, STUDY_MAX_MODEL_LEN))
    extra_args = model_env("EXTRA_ARGS", "")
    overridden = extra_args_overriding_held_constants(extra_args)
    if overridden:
        res.nonconformant("EXTRA_ARGS overrides a held-constant serving flag (%s in "
                          "models.d/%s.env)" % (", ".join(overridden), args.model))

    # ---- hardware / price --------------------------------------------------
    hardware, gpu = resolve_hardware(res, node_count)
    price = resolve_price(repo, hardware["instance_type"], node_count, res)
    # A loopback endpoint is the self-hosted case: the model is billed by the instance hour.
    # Anything else is a hosted API billed per token (CONTRACTS.md §2.2 "price").
    price["billing_mode"] = "instance_hours" if endpoint_is_loopback(args.endpoint) else "per_token"

    # ---- inference ---------------------------------------------------------
    inference = dict(inference_cfg)
    inference["endpoint"] = args.endpoint
    inference["passes"] = args.passes
    inference["concurrency"] = args.concurrency
    inference["task_timeout_s"] = args.task_timeout
    inference["max_iters"] = args.max_iters
    if args.max_iters != INFERENCE_DEFAULTS["max_iters"]:
        res.nonconformant("--max-iters %d is not the held-constant budget %d"
                          % (args.max_iters, INFERENCE_DEFAULTS["max_iters"]))
    # Every knob that can change a verdict is a held constant (CONTRACTS.md §0.1). Checking
    # only max_iters would let an edited agent_config.json produce a run that claims
    # flags.nonconformant = false while running a different harness — which is precisely
    # what the flag exists to prevent. `concurrency` and `passes` are throughput knobs and
    # are deliberately excluded: they cannot change what a single attempt does.
    _THROUGHPUT_KNOBS = ("concurrency", "passes", "endpoint")
    for _key, _want in sorted(INFERENCE_DEFAULTS.items()):
        if _key in _THROUGHPUT_KNOBS or _key == "max_iters":
            continue
        _got = inference.get(_key)
        if _got != _want:
            res.nonconformant("inference.%s is %r, not the held-constant %r"
                              % (_key, _got, _want))

    # ---- ci ----------------------------------------------------------------
    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    ci = {
        "provider": "github-actions" if in_ci else "local",
        "workflow": os.environ.get("GITHUB_WORKFLOW") or None,
        "github_run_id": os.environ.get("GITHUB_RUN_ID") or None,
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT") or None,
        "actor": os.environ.get("GITHUB_ACTOR") or os.environ.get("USER") or None,
        "triggered_at": os.environ.get("HARNESS_TRIGGERED_AT") or None,
    }

    # runtime.vllm_argv is the only record of how the server was actually launched.
    # modelctl writes it to $STATE_DIR/vllm-argv and run.sh copies it into the run dir.
    vllm_argv = read_vllm_argv(run_dir)
    if vllm_argv is None and strict:
        res.provenance("runtime.vllm_argv unresolved — $STATE_DIR/vllm-argv was not "
                       "written by modelctl (was the server started another way?)")

    consent_class = SUITES[args.suite][2]
    exploratory = bool(args.instance) or args.limit is not None

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": args.run_id,
        "run_group_id": args.run_group_id or None,
        "status": args.status,
        "created_at": created_at,
        "harness": {
            "version": harness_version,
            "repo_git_sha": git_sha,
            "repo_git_describe": git_describe,
            "repo_dirty": git_dirty,
            "invocation": invocation,
            "result_schema": RESULT_SCHEMA,
            "prompt_template_id": template_id,
            "prompt_dir_sha256": prompt_dir_sha,
            "agent_config_sha256": agent_config_sha,
            "adapter": adapter_rel,
            "adapter_version": adapter_version,
            "adapter_sha256": adapter_sha,
            "adapters_dir_sha256": adapters_dir_sha,
            "environment_digest": env_digest,
        },
        "suite": {
            "name": args.suite,
            "seed_file": repo_rel(seed_path),
            "seed_file_sha256": sha256_file(seed_path),
            "selection_seed": seed_blob["selection"]["seed"],
            "selection_method": seed_blob["selection"]["method"],
            "instance_count": len(ids),
            "instance_ids": ids,
            "instance_ids_sha256": id_set_sha256(ids),
            "partitions_file": repo_rel(partitions_path),
            "partitions_sha256": sha256_file(partitions_path),
        },
        "model": {
            "name": args.model,
            "served_model_name": res.required("model.served_model_name", args.served_model_name),
            "hf_repo": hf_repo,
            "weight_revision": weight_revision,
            "weight_revision_source": weight_revision_source,
            "weight_digest": weight_digest,
            "weight_file_count": weight_files,
            "weight_bytes": weight_bytes,
            "weights_dir": str(weights_dir),
            "quantization": read_quantization(weights_dir),
            "model_env_sha256": sha256_file(model_env_file),
        },
        "runtime": {
            "vllm_version": vllm_version,
            "vllm_dist_digest": vllm_dist_digest(vllm_version) if vllm_version else None,
            "vllm_docker_image": docker_image,
            "vllm_docker_image_digest": docker_digest,
            "python_version": platform.python_version(),
            "torch_version": dist_version("torch"),
            "transformers_version": dist_version("transformers"),
            "nvidia_driver": gpu["driver"],
            "cuda_runtime": gpu["cuda"],
            "pip_freeze_sha256": sha256_file(pip_freeze) if pip_freeze.is_file() else None,
            "requirements_lock_sha256": lock_sha,
            "tensor_parallel_size": as_int("TP", 1),
            "pipeline_parallel_size": as_int("PP", 1),
            "max_model_len": max_model_len,
            "extra_args": extra_args,
            "multinode": multinode,
            "vllm_argv": vllm_argv,
        },
        "inference": inference,
        "hardware": hardware,
        "ci": ci,
        "price": price,
        "timing": {
            "started_at": created_at,
            "ended_at": None,
            # wall_clock_s spans started_at..ended_at and therefore includes the idle gap
            # between a run and its --resume; active_wall_clock_s accumulates one span per
            # invocation and is what the headline cost uses (CONTRACTS.md §2.2, §8).
            "wall_clock_s": None,
            "active_wall_clock_s": None,
            "invocation_count": 1,
            "attempts_planned": len(ids) * args.passes,
            "attempts_written": 0,
        },
        "flags": {
            "exploratory": exploratory,
            "truncated": truncated,
            # Two different things, never conflated (CONTRACTS.md §2.2):
            "nonconformant": bool(res.nonconformant_reasons),
            "nonconformant_reasons": list(res.nonconformant_reasons),
            "provenance_incomplete": bool(res.provenance_reasons),
            "provenance_incomplete_reasons": list(res.provenance_reasons),
            "grading_degraded": False,
            "resumed_from": args.resumed_from or None,
            "consent_class": consent_class,
        },
        # notes is written once and never changes (CONTRACTS.md §2.3)
        "notes": "; ".join(res.reasons),
    }

    if strict and res.unresolved:
        manifest["status"] = "failed"
    write_json(run_dir / "run-manifest.json", manifest)

    if strict and res.unresolved:
        die("REQUIRED provenance could not be resolved: %s" % ", ".join(res.unresolved), 3)
    return 0


# ------------------------------------------------------------------------- finalize

def set_flag(flags, bool_key: str, reasons_key: str, reasons=()) -> None:
    """Write-once boolean plus a de-duplicated reason list (CONTRACTS.md §2.2).

    Never clears: a flag raised by the pre-run manifest write survives finalization.
    """
    existing = flags.get(reasons_key)
    if not isinstance(existing, list):
        existing = []
    for reason in reasons:
        if reason and reason not in existing:
            existing.append(reason)
    flags[reasons_key] = existing
    flags[bool_key] = True


def cmd_finalize(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    path = run_dir / "run-manifest.json"
    manifest = None
    if path.is_file():
        try:
            manifest = read_json(path)
        except ValueError as exc:
            warn("%s is not valid JSON (%s) — leaving it alone" % (path, exc))
    else:
        warn("no run-manifest.json under %s — writing run-status.json only" % run_dir)

    if manifest is None:
        write_json(run_dir / "run-status.json", {
            "status": args.status,
            "exit_code": args.exit_code,
            "attempts_planned": None,
            "attempts_written": args.attempts_written,
            "started_at": args.started_at or None,
            "ended_at": args.ended_at or utcnow(),
        })
        return 0

    started = manifest["timing"]["started_at"] or args.started_at
    ended = args.ended_at or utcnow()
    wall = None
    if started:
        wall = max(0, int((parse_ts(ended) - parse_ts(started)).total_seconds()))

    # A --resume invocation reuses the manifest verbatim, so timing.started_at is the
    # ORIGINAL start and wall_clock_s spans the idle gap between the two invocations —
    # hours of it, in the normal case where a run is resumed the next morning. That number
    # must never reach the headline cost, so active wall clock is accumulated per
    # invocation and CONTRACTS.md §8 is computed from it.
    this_started = args.started_at or started
    span = 0
    if this_started:
        span = max(0, int((parse_ts(ended) - parse_ts(this_started)).total_seconds()))
    prior = manifest["timing"].get("active_wall_clock_s")
    prior = int(prior) if isinstance(prior, (int, float)) and not isinstance(prior, bool) else 0
    invocations = manifest["timing"].get("invocation_count")
    invocations = int(invocations) if isinstance(invocations, int) and invocations > 0 else 1

    manifest["status"] = args.status
    manifest["timing"]["started_at"] = started
    manifest["timing"]["ended_at"] = ended
    manifest["timing"]["wall_clock_s"] = wall
    if args.resumed_from:
        manifest["timing"]["active_wall_clock_s"] = prior + span
        manifest["timing"]["invocation_count"] = invocations + 1
    else:
        # First (and normally only) invocation: active == this span, and finalizing twice
        # cannot inflate it.
        manifest["timing"]["active_wall_clock_s"] = span
        manifest["timing"]["invocation_count"] = 1
    manifest["timing"]["attempts_written"] = args.attempts_written

    flags = manifest.setdefault("flags", {})
    flags.setdefault("nonconformant", False)
    flags.setdefault("nonconformant_reasons", [])
    flags.setdefault("provenance_incomplete", False)
    flags.setdefault("provenance_incomplete_reasons", [])
    if args.grading_degraded:
        flags["grading_degraded"] = True
    if args.nonconformant or args.nonconformant_reason:
        set_flag(flags, "nonconformant", "nonconformant_reasons", args.nonconformant_reason)
    if args.provenance_incomplete or args.provenance_reason:
        set_flag(flags, "provenance_incomplete", "provenance_incomplete_reasons",
                 args.provenance_reason)
    if args.resumed_from:
        # §2.2: the run this invocation continued. An in-place --resume reuses the run id,
        # so this equals run_id — what matters downstream is that it is not null, because
        # that is the marker that wall_clock_s includes idle time.
        flags["resumed_from"] = args.resumed_from
    write_json(path, manifest)

    write_json(run_dir / "run-status.json", {
        "status": args.status,
        "exit_code": args.exit_code,
        "attempts_planned": manifest["timing"]["attempts_planned"],
        "attempts_written": args.attempts_written,
        "started_at": started,
        "ended_at": ended,
    })
    return 0


# ----------------------------------------------------------------- grading preflight

# What each suite's grade() needs on this host. Checked BEFORE the first model call, so a
# run cannot burn hours of GPU budget only to discover at the first grade() that it can
# never produce a verdict. Refined at run time from the adapter's own SuiteSpec.
GRADING_DEPS = {
    "swebench-verified": {"binaries": ("docker",), "modules": ("swebench",)},
    "swebench-pro": {"binaries": ("docker",), "modules": ("swebench",)},
    "agenttask": {"binaries": ("git",), "modules": ()},
}

BINARY_HINTS = {
    "docker": "install Docker and make sure `docker version` succeeds as this user — the "
              "SWE-bench evaluation harness builds one container per instance",
    "git": "install git — the agenttask grader replays tests inside a git workspace",
}
MODULE_HINTS = {
    "swebench": "python3 -m pip install swebench   (the official evaluation harness)",
}

SKIP_GRADING_PREFLIGHT = "HARNESS_SKIP_GRADING_PREFLIGHT"


def module_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 - a broken or missing parent package means "no"
        return False


def cmd_grading_preflight(args) -> int:
    """Refuse to start a run this host could never grade (CONTRACTS.md §1.3 exit 3).

    Calls the selected adapter's environment_digest() — the same call the manifest makes —
    and checks the binaries and modules grade() will reach for. Prints the digest as
    KEY=VALUE on stdout; every human word goes to stderr.
    """
    import shutil

    repo = Path(args.repo).resolve()
    suite = args.suite

    if os.environ.get(SKIP_GRADING_PREFLIGHT) == "1":
        warn("grading preflight skipped (%s=1) — a missing grader will surface as "
             "INFRA_GRADER / INFRA_SANDBOX on every attempt instead" % SKIP_GRADING_PREFLIGHT)
        sys.stdout.write("PREFLIGHT_SKIPPED=1\n")
        return 0

    try:
        mod = load_adapter_module(repo, suite)
    except Exception as exc:  # noqa: BLE001
        die("cannot import the %s adapter (%s: %s) — grading is impossible, so nothing "
            "was executed" % (suite, type(exc).__name__, exc), 3)

    deps = GRADING_DEPS.get(suite) or {}
    binaries = list(deps.get("binaries") or ())
    modules = list(deps.get("modules") or ())
    missing = []  # (what is missing, how to fix it)

    # The adapter's own SuiteSpec wins over the table above.
    spec = getattr(mod, "SPEC", None)
    eval_module = getattr(spec, "eval_module", None) if spec is not None else None
    if eval_module:
        modules = [str(eval_module).split(".")[0]]
        override = None
        env_fn = getattr(spec, "env", None)
        if callable(env_fn):
            try:
                override = env_fn("EVAL_CMD")
            except Exception:  # noqa: BLE001
                override = None
        if override:
            # HARNESS_EVAL_CMD[_<SUITE>] replaces `python -m <eval_module>` wholesale.
            modules = []
            import shlex

            parts = shlex.split(override)
            exe = parts[0] if parts else ""
            if not exe or (shutil.which(exe) is None and not os.path.exists(exe)):
                missing.append(("the HARNESS_EVAL_CMD override %r is not executable" % exe,
                                "point HARNESS_EVAL_CMD at a real evaluation entry point"))

    for binary in binaries:
        if shutil.which(binary) is None:
            missing.append(("%s (not on PATH)" % binary,
                            BINARY_HINTS.get(binary, "install %s" % binary)))
    for module in modules:
        if not module_available(module):
            missing.append(("the %s python module (not importable)" % module,
                            MODULE_HINTS.get(module, "python3 -m pip install %s" % module)))

    if not callable(getattr(mod, "grade", None)):
        missing.append(("%s.grade()" % suite, "the adapter must expose grade() (§5)"))

    # An adapter may declare its own host requirements — the static GRADING_DEPS table
    # cannot know, for example, that agenttask grades by running pytest in-process. Without
    # this an adapter whose runner is absent passes preflight and then grades every attempt
    # TESTS_FAIL 0/N, which publishes as a model result rather than as a broken host.
    reqs_fn = getattr(mod, "grading_requirements", None)
    if callable(reqs_fn):
        try:
            for name, present, hint in reqs_fn():
                if not present:
                    missing.append((name, hint))
        except Exception as exc:  # noqa: BLE001
            missing.append(("%s.grading_requirements() raised %s: %s"
                            % (suite, type(exc).__name__, exc),
                            "the adapter could not report its host requirements"))

    digest = None
    func = getattr(mod, "environment_digest", None)
    if not callable(func):
        missing.append(("%s.environment_digest()" % suite,
                        "the adapter must expose environment_digest() (§5)"))
    else:
        try:
            digest = func()
        except Exception as exc:  # noqa: BLE001
            missing.append(("environment_digest() raised %s: %s" % (type(exc).__name__, exc),
                            "the grading environment could not be identified"))

    if digest:
        sys.stdout.write("PREFLIGHT_ENVIRONMENT_DIGEST=%s\n" % digest)
    if missing:
        for what, hint in missing:
            warn("grading preflight: MISSING %s" % what)
            warn("grading preflight:   fix: %s" % hint)
        sys.stdout.write("PREFLIGHT_OK=0\n")
        die("suite '%s' cannot be graded on this host — missing: %s. Nothing was "
            "executed; fix the dependency or set %s=1 to run anyway."
            % (suite, "; ".join(what for what, _hint in missing), SKIP_GRADING_PREFLIGHT), 3)
    sys.stdout.write("PREFLIGHT_OK=1\n")
    return 0


# --------------------------------------------------------------------- prompt-check

def cmd_prompt_check(args) -> int:
    """CONTRACTS.md §5.2: run.sh asserts the adapter's rendered template id against the
    manifest for the first task of every run, and exits 2 on mismatch."""
    repo = Path(args.repo).resolve()
    manifest = read_json(Path(args.manifest))
    suite = manifest["suite"]["name"]
    expected = manifest["harness"]["prompt_template_id"]
    seed_file = repo / manifest["suite"]["seed_file"]
    if not seed_file.is_file():
        seed_file = Path(manifest["suite"]["seed_file"])

    sys.path.insert(0, str(repo))
    try:
        import importlib

        try:
            mod = importlib.import_module("harness.adapters").get(suite)
        except Exception:  # noqa: BLE001
            name = Path(SUITES[suite][0]).stem
            mod = importlib.import_module("harness.adapters." + name)
        tasks = mod.load_tasks(seed_file)
        if not tasks:
            die("adapter for %s loaded 0 tasks from %s" % (suite, seed_file), 2)
        wanted = manifest["suite"]["instance_ids"][0]
        first = next((t for t in tasks if t.instance_id == wanted), tasks[0])
        prompt = mod.build_prompt(first)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        die("could not render a prompt via the %s adapter: %s" % (suite, exc), 2)
    finally:
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)

    if prompt.template_id != expected:
        die("harness-constant violation: %s renders template %r but the manifest records "
            "%r — every suite must use the same template (CONTRACTS.md §5.2)"
            % (suite, prompt.template_id, expected), 2)

    if args.write_preview:
        if manifest["flags"]["consent_class"] == "restricted":
            warn("consent_class=restricted — prompt-preview.txt must never be committed")
        out = Path(args.write_preview)
        out.write_text(
            "# template_id : %s\n# instance_id : %s\n# prompt_sha256: %s\n# tools       : %d\n"
            "\n----- system -----\n%s\n\n----- user -----\n%s\n"
            % (prompt.template_id, first.instance_id, prompt.prompt_sha256,
               len(prompt.tools), prompt.system, prompt.user),
            encoding="utf-8")
    sys.stdout.write("%s\n" % prompt.template_id)
    return 0


# ------------------------------------------------------------------- scan / missing

def read_records(run_dir: Path, run_id):
    """Yield (record, ok) for every line of results.jsonl."""
    path = run_dir / "results.jsonl"
    if not path.is_file():
        return [], 0, 0
    good, malformed, foreign = [], 0, 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if not isinstance(rec, dict):
                malformed += 1
                continue
            if run_id and rec.get("run_id") != run_id:
                foreign += 1
                continue
            good.append(rec)
    return good, malformed, foreign


def cmd_scan(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    recs, malformed, foreign = read_records(run_dir, args.run_id)
    seen = set()
    resolved = infra_grader = infra_unknown = infra_any = 0
    server = sandbox = 0
    retryable = 0  # records `missing` would re-run — a run holding any is not complete
    # trailing_server: consecutive SERVER_* at the END of the record stream — the signature
    # of a vLLM process that died mid-run, as opposed to sporadic transient failures.
    trailing_server = 0
    for rec in recs:
        seen.add((rec.get("instance_id"), rec.get("pass_idx")))
        code = rec.get("error_code") or ""
        if rec.get("resolved") is True:
            resolved += 1
        if code == "INFRA_GRADER":
            infra_grader += 1
        if code == "INFRA_UNKNOWN":
            infra_unknown += 1
        if code.startswith("INFRA_"):
            infra_any += 1
        if code.startswith("SERVER_"):
            server += 1
            trailing_server += 1
        else:
            trailing_server = 0
        if code == "INFRA_SANDBOX":
            sandbox += 1
        if code in RETRYABLE_ERROR_CODES:
            retryable += 1
    out = [
        ("SCAN_RECORDS", len(recs)),
        ("SCAN_UNIQUE", len(seen)),
        ("SCAN_RESOLVED", resolved),
        ("SCAN_INFRA_GRADER", infra_grader),
        ("SCAN_INFRA_UNKNOWN", infra_unknown),
        ("SCAN_SERVER", server),
        ("SCAN_TRAILING_SERVER", trailing_server),
        ("SCAN_RETRYABLE", retryable),
        ("SCAN_INFRA_SANDBOX", sandbox),
        ("SCAN_ATTEMPTS_SCORED", len(recs) - infra_any),
        ("SCAN_MALFORMED", malformed),
        ("SCAN_FOREIGN", foreign),
    ]
    sys.stdout.write("".join("%s=%d\n" % kv for kv in out))
    return 0


def cmd_missing(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest = read_json(run_dir / "run-manifest.json")
    recs, _m, _f = read_records(run_dir, manifest["run_id"])
    # An INFRA_HOST record was never scored (the host went away mid-attempt): it is
    # retryable, so its instance is still missing. run.sh prunes those records before the
    # pass re-runs (prune-retryable), which keeps §3.1 "exactly one record per attempt".
    done = set(r.get("instance_id") for r in recs
               if r.get("pass_idx") == args.pass_idx
               and (r.get("error_code") or "") not in RETRYABLE_ERROR_CODES)
    missing = [i for i in manifest["suite"]["instance_ids"] if i not in done]
    if args.out:
        Path(args.out).write_text("".join(i + "\n" for i in missing), encoding="utf-8")
    sys.stdout.write("MISSING_COUNT=%d\n" % len(missing))
    return 0


def cmd_prune_retryable(args) -> int:
    """Rewrite results.jsonl without this pass's INFRA_HOST records (seam S1).

    Atomic (tmp + os.replace); every other line — other passes, other error codes,
    malformed or foreign lines — is preserved byte for byte. Prints PRUNED=<count>.
    """
    run_dir = Path(args.run_dir).resolve()
    path = run_dir / "results.jsonl"
    if not path.is_file():
        sys.stdout.write("PRUNED=0\n")
        return 0
    run_id = None
    try:
        run_id = read_json(run_dir / "run-manifest.json").get("run_id")
    except (OSError, ValueError, AttributeError):
        pass
    kept = []
    pruned = 0
    with open(path, "rb") as fh:
        for raw in fh:
            drop = False
            try:
                rec = json.loads(raw.decode("utf-8"))
            except ValueError:
                rec = None
            if isinstance(rec, dict) \
                    and rec.get("pass_idx") == args.pass_idx \
                    and (rec.get("error_code") or "") in RETRYABLE_ERROR_CODES \
                    and (run_id is None or rec.get("run_id") == run_id):
                drop = True
            if drop:
                pruned += 1
            else:
                kept.append(raw)
    if pruned:
        tmp = str(path) + ".prune.tmp"
        with open(tmp, "wb") as fh:
            fh.writelines(kept)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, str(path))
    sys.stdout.write("PRUNED=%d\n" % pruned)
    return 0


# ------------------------------------------------------------------------ checksums

def cmd_checksums(args) -> int:
    run_dir = Path(args.run_dir).resolve()
    out = run_dir / "SHA256SUMS"
    rows = []
    for dirpath, dirnames, filenames in os.walk(str(run_dir), followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not os.path.islink(os.path.join(dirpath, d)))
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, str(run_dir)).replace(os.sep, "/")
            if rel == "SHA256SUMS" or os.path.islink(full):
                continue
            rows.append((rel, sha256_file(full)))
    rows.sort(key=lambda r: r[0].encode("utf-8"))
    with open(str(out) + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        for rel, h in rows:
            fh.write("%s  %s\n" % (h, rel))
    os.replace(str(out) + ".tmp", str(out))
    sys.stdout.write("SHA256SUMS_FILES=%d\n" % len(rows))
    return 0


# ------------------------------------------------------------------------------ get

def cmd_get(args) -> int:
    node = read_json(Path(args.file))
    for part in args.path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    if isinstance(node, bool):
        sys.stdout.write("true\n" if node else "false\n")
    elif node is None:
        sys.stdout.write("\n")
    elif isinstance(node, (dict, list)):
        sys.stdout.write(json.dumps(node, sort_keys=True) + "\n")
    else:
        sys.stdout.write("%s\n" % node)
    return 0


# ----------------------------------------------------------------------- dir-digest

def cmd_dir_digest(args) -> int:
    root = Path(args.dir).resolve()
    if not root.is_dir():
        die("%s is not a directory" % root, 2)
    digest, count, total = dir_digest(root, symlinks_fatal=args.symlinks_fatal,
                                      workers=args.workers)
    if args.verbose:
        sys.stderr.write("==> manifest: %d file(s), %d byte(s), workers=%s\n"
                         % (count, total, args.workers or DIGEST_WORKERS))
    sys.stdout.write(digest + "\n")
    return 0


# ----------------------------------------------------------------------------- main

def main(argv) -> int:
    ap = argparse.ArgumentParser(prog="manifest.py", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    b.add_argument("--run-dir", required=True)
    b.add_argument("--run-id", required=True)
    b.add_argument("--run-group-id", default="")
    b.add_argument("--model", required=True)
    b.add_argument("--suite", required=True, choices=sorted(SUITES))
    b.add_argument("--seed-file", required=True)
    b.add_argument("--partitions", required=True)
    b.add_argument("--endpoint", required=True)
    b.add_argument("--weights-dir", required=True)
    b.add_argument("--passes", type=int, required=True)
    b.add_argument("--concurrency", type=int, required=True)
    b.add_argument("--task-timeout", type=int, required=True)
    b.add_argument("--max-iters", type=int, required=True)
    b.add_argument("--served-model-name", default="")
    b.add_argument("--invocation-file", default="")
    b.add_argument("--created-at", default="")
    b.add_argument("--status", default="running")
    b.add_argument("--mode", default="exec", choices=["exec", "dry-run", "manifest-only"])
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--instance", action="append", default=[])
    b.add_argument("--resumed-from", default="")
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("finalize")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--status", required=True)
    f.add_argument("--exit-code", type=int, required=True)
    f.add_argument("--attempts-written", type=int, required=True)
    f.add_argument("--started-at", default="")
    f.add_argument("--ended-at", default="")
    f.add_argument("--grading-degraded", action="store_true")
    f.add_argument("--nonconformant", action="store_true")
    f.add_argument("--nonconformant-reason", action="append", default=[])
    f.add_argument("--provenance-incomplete", action="store_true")
    f.add_argument("--provenance-reason", action="append", default=[])
    f.add_argument("--resumed-from", default="")
    f.set_defaults(func=cmd_finalize)

    gp = sub.add_parser("grading-preflight")
    gp.add_argument("--repo", required=True)
    gp.add_argument("--suite", required=True, choices=sorted(SUITES))
    gp.set_defaults(func=cmd_grading_preflight)

    p = sub.add_parser("prompt-check")
    p.add_argument("--repo", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--write-preview", default="")
    p.set_defaults(func=cmd_prompt_check)

    s = sub.add_parser("scan")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--run-id", default="")
    s.set_defaults(func=cmd_scan)

    m = sub.add_parser("missing")
    m.add_argument("--run-dir", required=True)
    m.add_argument("--pass-idx", type=int, required=True)
    m.add_argument("--out", default="")
    m.set_defaults(func=cmd_missing)

    pr = sub.add_parser("prune-retryable")
    pr.add_argument("--run-dir", required=True)
    pr.add_argument("--pass-idx", type=int, required=True)
    pr.set_defaults(func=cmd_prune_retryable)

    c = sub.add_parser("checksums")
    c.add_argument("--run-dir", required=True)
    c.set_defaults(func=cmd_checksums)

    g = sub.add_parser("get")
    g.add_argument("file")
    g.add_argument("path")
    g.set_defaults(func=cmd_get)

    dd = sub.add_parser("dir-digest")
    dd.add_argument("dir")
    dd.add_argument("--workers", type=int, default=None,
                    help="hashing threads (1 = serial reference; default %d)" % DIGEST_WORKERS)
    dd.add_argument("--symlinks-fatal", action="store_true")
    dd.add_argument("--verbose", action="store_true")
    dd.set_defaults(func=cmd_dir_digest)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
