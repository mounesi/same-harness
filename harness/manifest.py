#!/usr/bin/env python3
# harness/manifest.py — provenance collector and run-directory bookkeeper for harness/run.sh.
#
#   manifest.py build        ...  resolve every provenance field, write run-manifest.json
#   manifest.py finalize     ...  rewrite status/timing/flags, write run-status.json
#   manifest.py prompt-check ...  assert the adapter renders the manifest's template id
#   manifest.py scan         ...  attempt + failure histogram of results.jsonl (KEY=VALUE)
#   manifest.py missing      ...  instance ids with no record for a pass (for --resume)
#   manifest.py checksums    ...  write SHA256SUMS over a run directory
#   manifest.py get          ...  print one dotted field of a JSON file
#
# Implements docs/CONTRACTS.md §2 (run-manifest/v1) and §2.4 (directory digest).
# Python 3.11 stdlib only. Never touches the network unless HARNESS_ALLOW_NETWORK=1.
#
# Exit codes match run.sh: 0 ok, 2 config error, 3 unresolved REQUIRED provenance.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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

UNRESOLVED = "unresolved"


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


def dir_digest(root: Path, symlinks_fatal: bool = False):
    """CONTRACTS.md §2.4. Returns (hexdigest, file_count, total_bytes)."""
    files, links = walk_files(root)
    if links and symlinks_fatal:
        raise SymlinkError("%s contains symlinks (%s ...) — weights must be real files"
                           % (root, links[0]))
    if links:
        warn("%s: skipped %d symlink(s) per §2.4" % (root, len(links)))
    pairs = []
    total = 0
    for rel, full in files:
        pairs.append((rel, sha256_file(full)))
        total += os.path.getsize(full)
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


def cached_dir_digest(root: Path, cache_key: str):
    """Digest a very large tree once. Cache lives outside the tree so it cannot
    perturb the digest it describes."""
    cache = Path.home() / ".harness" / "weight-digest" / (cache_key + ".json")
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
    """Collects values, remembering which REQUIRED ones could not be resolved and
    which resolutions cost us comparability."""

    def __init__(self, strict: bool):
        self.strict = strict
        self.unresolved = []
        self.reasons = []

    def nonconformant(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
            warn("nonconformant: " + reason)

    def required(self, name: str, value, sentinel=UNRESOLVED):
        if value in (None, "", []):
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

    if not instance_type:
        res.nonconformant("hardware.instance_type unresolved")
    if not instance_id:
        res.nonconformant("hardware.lambda_instance_id unresolved — cost cannot be "
                          "reconciled against Lambda billing")
    if not region:
        res.nonconformant("hardware.region unresolved")

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
        res.nonconformant("price unresolved — no instance type")
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
                    regions = [] if m.group(3).strip() in ("", "-") else \
                        [r.strip() for r in m.group(3).split(",") if r.strip()]
                    cents = int(round(float(m.group(2)) * 100))
                    return finish("lambdactl-types", utcnow(), cents, regions)
        warn("./lambdactl types did not report %s" % instance_type)

    fallback = repo / "pricing" / "fallback-prices.json"
    if fallback.is_file():
        got = from_snapshot(fallback, "static-fallback")
        if got:
            res.nonconformant("price came from pricing/fallback-prices.json, not a live snapshot")
            return got

    res.nonconformant("price unresolved for %s" % instance_type)
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
    recorded = blob.get("instance_ids_sha256")
    if recorded and recorded != id_set_sha256(ids):
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
    return blob


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
    load_partitions(partitions_path)

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
        res.nonconformant("weight digest skipped (HARNESS_SKIP_WEIGHT_DIGEST=1)")
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
    weight_digest = res.required("model.weight_digest", weight_digest)

    weight_revision, weight_revision_source = resolve_weight_revision(weights_dir, hf_repo, res)

    # ---- runtime -----------------------------------------------------------
    vllm_version = dist_version("vllm")
    if not vllm_version:
        res.nonconformant("vllm is not importable here — runtime.vllm_version unresolved")
    docker_image = model_env("VLLM_DOCKER_IMAGE").strip() or None
    docker_digest = None
    if docker_image:
        docker_digest = sh(["docker", "inspect", "--format", "{{index .RepoDigests 0}}", docker_image])
        if not docker_digest:
            res.required("runtime.vllm_docker_image_digest", None, None)

    pip_freeze = run_dir / "env" / "pip-freeze.txt"
    lock = repo / "harness" / "requirements.lock"
    multinode = model_env("MULTINODE", "0") == "1"
    node_count = 2 if multinode else 1

    def as_int(name, default):
        raw = model_env(name, "").strip()
        try:
            return int(raw)
        except ValueError:
            return default

    # ---- hardware / price --------------------------------------------------
    hardware, gpu = resolve_hardware(res, node_count)
    price = resolve_price(repo, hardware["instance_type"], node_count, res)

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
            "requirements_lock_sha256": sha256_file(lock) if lock.is_file() else None,
            "tensor_parallel_size": as_int("TP", 1),
            "pipeline_parallel_size": as_int("PP", 1),
            "max_model_len": as_int("MAX_MODEL_LEN", 262144),
            "extra_args": model_env("EXTRA_ARGS", ""),
            "multinode": multinode,
            "vllm_argv": read_vllm_argv(run_dir),
        },
        "inference": inference,
        "hardware": hardware,
        "ci": ci,
        "price": price,
        "timing": {
            "started_at": created_at,
            "ended_at": None,
            "wall_clock_s": None,
            "attempts_planned": len(ids) * args.passes,
            "attempts_written": 0,
        },
        "flags": {
            "exploratory": exploratory,
            "truncated": truncated,
            "nonconformant": bool(res.reasons),
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

    manifest["status"] = args.status
    manifest["timing"]["started_at"] = started
    manifest["timing"]["ended_at"] = ended
    manifest["timing"]["wall_clock_s"] = wall
    manifest["timing"]["attempts_written"] = args.attempts_written
    if args.grading_degraded:
        manifest["flags"]["grading_degraded"] = True
    if args.nonconformant:
        manifest["flags"]["nonconformant"] = True  # set-only, never cleared
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
    out = [
        ("SCAN_RECORDS", len(recs)),
        ("SCAN_UNIQUE", len(seen)),
        ("SCAN_RESOLVED", resolved),
        ("SCAN_INFRA_GRADER", infra_grader),
        ("SCAN_INFRA_UNKNOWN", infra_unknown),
        ("SCAN_SERVER", server),
        ("SCAN_TRAILING_SERVER", trailing_server),
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
    done = set(r.get("instance_id") for r in recs if r.get("pass_idx") == args.pass_idx)
    missing = [i for i in manifest["suite"]["instance_ids"] if i not in done]
    if args.out:
        Path(args.out).write_text("".join(i + "\n" for i in missing), encoding="utf-8")
    sys.stdout.write("MISSING_COUNT=%d\n" % len(missing))
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
    f.set_defaults(func=cmd_finalize)

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

    c = sub.add_parser("checksums")
    c.add_argument("--run-dir", required=True)
    c.set_defaults(func=cmd_checksums)

    g = sub.add_parser("get")
    g.add_argument("file")
    g.add_argument("path")
    g.set_defaults(func=cmd_get)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
