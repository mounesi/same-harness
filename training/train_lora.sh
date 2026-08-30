#!/usr/bin/env bash
# train_lora.sh — 8xH100 LoRA / QLoRA launcher for qwen3-coder-next (AI-P153 phase 2).
# Config-driven, and it writes an experiment manifest BEFORE training starts, exactly the way
# harness/run.sh writes a run manifest: an experiment without a manifest is not an experiment.
#
#   ./training/train_lora.sh doctor                       GPUs, trainer deps, dataset sanity
#   ./training/train_lora.sh config   <config.yaml>       print the resolved config as JSON
#   ./training/train_lora.sh manifest <config.yaml>       resolve + write the experiment manifest only
#   ./training/train_lora.sh run      <config.yaml>       manifest, then launch the training job
#
# Options (after the config path):
#   --no-weight-digest   skip the base-model content digest (fast, sets flags.nonconformant)
#   --dry-run            with `run`: print the launch command instead of executing it
#   --output-dir DIR     override runtime.output_dir from the config
#
# Env:
#   WEIGHTS_DIR   where model weights live (default: /persistent/models, else ~/models) — matches modelctl
#   TORCHRUN_BIN  torchrun executable (default: torchrun)
#
# The leakage guard is enforced here too: the dataset's dataset-manifest.json must carry the same
# final_holdout_sha256 that is compiled into training/build_dataset.py, and its split must be
# train or dev. See docs/CONTRACTS.md §6.2 and training/README.md.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$BASE_DIR/training"
MODELS_D="$BASE_DIR/models.d"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

if [[ -z "${WEIGHTS_DIR:-}" ]]; then
  if [[ -d /persistent ]]; then WEIGHTS_DIR=/persistent/models; else WEIGHTS_DIR="$HOME/models"; fi
fi

die()  { echo "error: $*" >&2; exit 2; }
guard(){ echo "LEAKAGE GUARD: $*" >&2; exit 3; }
info() { echo "==> $*" >&2; }
usage() { sed -n '2,24p' "$0"; exit 2; }

# ------------------------------------------------------------ config parse ---
# Deliberately a small YAML subset (two levels, scalars and inline lists) so training configs
# stay readable and this launcher stays stdlib-only. Anything fancier belongs in the trainer's
# own config, not here.
resolve_config() { # <config.yaml> <out.json>
  python3 - "$1" "$2" <<'PY'
import json, re, sys

src, dst = sys.argv[1], sys.argv[2]

def strip_comment(line: str) -> str:
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()

def coerce(raw: str):
    v = raw.strip()
    if not v:
        return ""
    if v[0] in "\"'" and v[-1] == v[0] and len(v) > 1:
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [coerce(x) for x in inner.split(",")] if inner else []
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

flat, section = {}, None
for lineno, raw in enumerate(open(src, encoding="utf-8"), 1):
    line = strip_comment(raw.rstrip("\n"))
    if not line.strip():
        continue
    indent = len(line) - len(line.lstrip(" "))
    if line.lstrip().startswith("-"):
        sys.exit(f"{src}:{lineno}: block lists are not supported — use [a, b] inline")
    if ":" not in line:
        sys.exit(f"{src}:{lineno}: expected 'key: value'")
    key, _, value = line.lstrip().partition(":")
    key = key.strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        sys.exit(f"{src}:{lineno}: bad key {key!r}")
    if indent == 0:
        if value.strip() == "":
            section = key
            continue
        section = None
        flat[key] = coerce(value)
    else:
        if section is None:
            sys.exit(f"{src}:{lineno}: indented key {key!r} outside any section")
        flat[f"{section}.{key}"] = coerce(value)

json.dump(flat, open(dst, "w", encoding="utf-8"), indent=2, sort_keys=True)
open(dst, "a", encoding="utf-8").write("\n")
PY
}

cfg() { # <resolved.json> <dotted.key> [default]
  python3 - "$1" "$2" "${3-__REQUIRED__}" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
key, default = sys.argv[2], sys.argv[3]
if key in doc and doc[key] is not None and doc[key] != "":
    v = doc[key]
    print(",".join(str(x) for x in v) if isinstance(v, list) else v)
elif default == "__REQUIRED__":
    sys.exit(f"missing required config key: {key}")
else:
    print(default)
PY
}

# --------------------------------------------------------------- utilities ---

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# CONTRACTS §2.4 directory digest — identical algorithm to the run manifest's weight_digest.
dir_digest() { # <root> -> "<sha256hex> <file_count> <bytes>"
  python3 - "$1" <<'PY'
import hashlib, os, sys
root = os.path.abspath(sys.argv[1])
SKIP_DIRS = {".git", ".cache", "__pycache__"}
SKIP_FILES = {".DS_Store"}
pairs, nbytes = [], 0
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
    for name in sorted(filenames):
        if name in SKIP_FILES or name.endswith(".pyc"):
            continue
        full = os.path.join(dirpath, name)
        if os.path.islink(full):
            sys.exit(f"symlink in weights directory is an error: {full}")
        h = hashlib.sha256()
        with open(full, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        rel = os.path.relpath(full, root).replace(os.sep, "/")
        pairs.append((rel.encode("utf-8"), h.hexdigest()))
        nbytes += os.path.getsize(full)
pairs.sort(key=lambda p: p[0])
stream = b"".join(f"{h}  ".encode() + rel + b"\n" for rel, h in pairs)
print(f"{hashlib.sha256(stream).hexdigest()} {len(pairs)} {nbytes}")
PY
}

load_model_env() { # <model name> — same defaults modelctl uses
  local f="$MODELS_D/$1.env"
  [[ -f "$f" ]] || die "unknown base model '$1' — no $f (see ./modelctl list)"
  TP=1 PP=1 MAX_MODEL_LEN=262144 EXTRA_ARGS="" MULTINODE=0 VLLM_DOCKER_IMAGE="" HF_REVISION=""
  # shellcheck source=/dev/null
  source "$f"
  [[ -n "${HF_REPO:-}" ]] || die "$f does not define HF_REPO"
  MODEL_DIR="$WEIGHTS_DIR/$1"
  MODEL_ENV_FILE="$f"
}

# ---------------------------------------------------------- leakage guard ----

assert_dataset_clean() { # <dataset-manifest.json>
  local dsm="$1"
  [[ -f "$dsm" ]] || die "dataset manifest not found: $dsm (build it with training/build_dataset.py)"
  local compiled ds_holdout split name
  compiled="$(python3 "$TRAINING_DIR/build_dataset.py" --print-holdout-constant)"
  [[ "$compiled" =~ ^[0-9a-f]{64}$ ]] || guard \
    "FINAL_HOLDOUT_SHA256 is not pinned in training/build_dataset.py (currently '$compiled') — freeze it before training: python3 training/build_dataset.py --freeze suites/partitions.json --write"
  ds_holdout="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['final_holdout_sha256'])" "$dsm")"
  split="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('split',''))" "$dsm")"
  name="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('dataset_name',''))" "$dsm")"
  case "$split" in
    train|dev) ;;
    *) guard "dataset '$name' declares split='$split' — only train and dev may be trained on" ;;
  esac
  [[ "$compiled" == "$ds_holdout" ]] || guard \
    "dataset '$name' was built against final_holdout_sha256=$ds_holdout but build_dataset.py is frozen against $compiled — the dataset and the guard disagree"
  info "leakage guard OK: dataset '$name' split=$split holdout=$compiled"
}

# ------------------------------------------------------------- experiment ----

new_experiment_id() { # <name>
  local ts hex
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  hex="$(python3 -c "import uuid; print(uuid.uuid4().hex[:6])")"
  echo "$1__${ts}__${hex}"
}

write_manifest() { # <resolved.json> <config path> <exp_dir> <exp_id> <weight_digest|-> <files...>
  python3 - "$@" <<'PY'
import datetime, hashlib, json, os, platform, subprocess, sys

resolved_path, config_path, exp_dir, exp_id, weight_digest = sys.argv[1:6]
cfg = json.load(open(resolved_path, encoding="utf-8"))

def get(key, default=None):
    v = cfg.get(key)
    return default if v is None or v == "" else v

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sh(*args):
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None

def version(dist):
    try:
        import importlib.metadata as md
        return md.version(dist)
    except Exception:
        return None

repo = os.environ.get("BASE_DIR", ".")
dirty = bool(sh("git", "-C", repo, "status", "--porcelain"))

train_file = get("data.train_file")
eval_file = get("data.eval_file")
dsm_path = get("data.dataset_manifest")
dsm = json.load(open(dsm_path, encoding="utf-8")) if dsm_path and os.path.exists(dsm_path) else {}

gpu_names = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader") or ""
gpu_list = [g.strip() for g in gpu_names.splitlines() if g.strip()]

wd, wcount, wbytes = (None, None, None)
if weight_digest != "-":
    parts = weight_digest.split()
    wd = "sha256:" + parts[0]
    wcount, wbytes = int(parts[1]), int(parts[2])

nonconformant = bool(dirty or wd is None)

manifest = {
    "schema": "experiment-manifest/v1",
    "experiment_id": exp_id,
    "status": "running",
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "phase": "AI-P153 phase-2 lora",
    "repo": {
        "git_sha": sh("git", "-C", repo, "rev-parse", "HEAD"),
        "git_describe": sh("git", "-C", repo, "describe", "--tags", "--always", "--dirty"),
        "dirty": dirty,
    },
    "config": {
        "path": os.path.abspath(config_path),
        "sha256": sha256_file(config_path),
        "resolved": cfg,
    },
    "base_model": {
        "name": get("experiment.base_model"),
        "hf_repo": os.environ.get("HF_REPO"),
        "hf_revision": os.environ.get("HF_REVISION") or "unresolved",
        "weights_dir": os.environ.get("MODEL_DIR"),
        "model_env_sha256": sha256_file(os.environ["MODEL_ENV_FILE"])
        if os.environ.get("MODEL_ENV_FILE")
        else None,
        "weight_digest": wd,
        "weight_file_count": wcount,
        "weight_bytes": wbytes,
        "quantization": get("lora.quantization", "none"),
    },
    "dataset": {
        "train_file": train_file,
        "train_sha256": sha256_file(train_file) if train_file and os.path.exists(train_file) else None,
        "train_records": sum(1 for _ in open(train_file, encoding="utf-8"))
        if train_file and os.path.exists(train_file)
        else None,
        "eval_file": eval_file,
        "eval_sha256": sha256_file(eval_file) if eval_file and os.path.exists(eval_file) else None,
        "dataset_manifest": dsm_path,
        "dataset_manifest_sha256": sha256_file(dsm_path) if dsm_path and os.path.exists(dsm_path) else None,
        "split": dsm.get("split"),
        "source_run_ids": dsm.get("source_run_ids"),
        "partitions_sha256": dsm.get("partitions_sha256"),
        "final_holdout_sha256": dsm.get("final_holdout_sha256"),
        "consent_class": dsm.get("consent_class"),
    },
    "hyperparams": {
        k: v for k, v in cfg.items() if k.startswith(("lora.", "train.", "data.max_seq_len"))
    },
    "hardware": {
        "gpu_model": gpu_list[0] if gpu_list else None,
        "gpu_count_visible": len(gpu_list) or None,
        "gpu_count_requested": get("runtime.gpus"),
        "nvidia_driver": next(
            iter((sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader") or "").splitlines()),
            None,
        ),
        "hostname": sh("hostname", "-f") or platform.node(),
    },
    "runtime": {
        "python_version": platform.python_version(),
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "peft_version": version("peft"),
        "trl_version": version("trl"),
        "accelerate_version": version("accelerate"),
        "bitsandbytes_version": version("bitsandbytes"),
        "launcher": get("runtime.launcher", "torchrun"),
        "entrypoint": get("runtime.entrypoint"),
    },
    "flags": {"nonconformant": nonconformant, "weight_digest_skipped": wd is None},
    "eval_rule": (
        "Report the tuned model on the untouched final_holdout partition only "
        "(training/README.md). Dev is for hyperparameter selection and nothing else."
    ),
    "timing": {"started_at": None, "ended_at": None, "wall_clock_s": None},
}
out = os.path.join(exp_dir, "experiment-manifest.json")
with open(out, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(out)
PY
}

finalize_manifest() { # <exp_dir> <status> <started_at epoch>
  python3 - "$1" "$2" "$3" <<'PY'
import datetime, json, os, sys, time
exp_dir, status, started = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = os.path.join(exp_dir, "experiment-manifest.json")
man = json.load(open(path, encoding="utf-8"))
def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
now = int(time.time())
man["status"] = status
man["timing"] = {"started_at": iso(started), "ended_at": iso(now), "wall_clock_s": now - started}
with open(path, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(man, indent=2, sort_keys=True) + "\n")
PY
}

# Emit the trainer's own config (axolotl-shaped by default; swap runtime.entrypoint and this
# block together if you move to another trainer).
write_trainer_config() { # <resolved.json> <exp_dir> <model_dir> -> path
  python3 - "$1" "$2" "$3" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
exp_dir, model_dir = sys.argv[2], sys.argv[3]

def g(key, default=None):
    v = cfg.get(key)
    return default if v is None or v == "" else v

quant = str(g("lora.quantization", "none")).lower()
lines = [
    "# generated by training/train_lora.sh — do not edit; edit the source config instead",
    f"base_model: {model_dir}",
    "model_type: AutoModelForCausalLM",
    "tokenizer_type: AutoTokenizer",
    "adapter: qlora" if quant == "nf4" else "adapter: lora",
    f"load_in_4bit: {'true' if quant == 'nf4' else 'false'}",
    "strict: false",
    "datasets:",
    f"  - path: {g('data.train_file')}",
    "    type: chat_template",
    "    field_messages: messages",
]
if g("data.eval_file"):
    lines += [
        "test_datasets:",
        f"  - path: {g('data.eval_file')}",
        "    type: chat_template",
        "    field_messages: messages",
        "    split: train",
    ]
lines += [
    f"sequence_len: {g('data.max_seq_len', 32768)}",
    "sample_packing: false",
    "pad_to_sequence_len: true",
    f"lora_r: {g('lora.r', 32)}",
    f"lora_alpha: {g('lora.alpha', 64)}",
    f"lora_dropout: {g('lora.dropout', 0.05)}",
    "lora_target_modules:",
]
targets = g("lora.target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
if not isinstance(targets, list):
    targets = str(targets).split(",")
for t in targets:
    t = str(t).strip()
    if t:
        lines.append(f"  - {t}")
lines += [
    f"num_epochs: {g('train.epochs', 2)}",
    f"micro_batch_size: {g('train.micro_batch_size', 1)}",
    f"gradient_accumulation_steps: {g('train.grad_accum', 8)}",
    f"learning_rate: {g('train.learning_rate', 0.0001)}",
    f"lr_scheduler: {g('train.lr_scheduler', 'cosine')}",
    f"warmup_ratio: {g('train.warmup_ratio', 0.03)}",
    f"weight_decay: {g('train.weight_decay', 0.0)}",
    f"optimizer: {g('train.optimizer', 'adamw_torch_fused')}",
    f"bf16: {str(bool(g('train.bf16', True))).lower()}",
    f"gradient_checkpointing: {str(bool(g('train.gradient_checkpointing', True))).lower()}",
    f"flash_attention: {str(bool(g('train.flash_attention', True))).lower()}",
    f"save_steps: {g('train.save_steps', 200)}",
    f"eval_steps: {g('train.eval_steps', 200)}",
    f"logging_steps: {g('train.logging_steps', 10)}",
    f"seed: {g('experiment.seed', 20260830)}",
    f"deepspeed: {g('train.deepspeed')}" if g("train.deepspeed") else "",
    f"output_dir: {os.path.join(exp_dir, 'adapter')}",
]
path = os.path.join(exp_dir, "trainer-config.yaml")
with open(path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(l for l in lines if l) + "\n")
print(path)
PY
}

# ---------------------------------------------------------------- commands ---

prepare() { # <config> [flags...] — sets EXP_DIR, EXP_ID, RESOLVED, TRAINER_CFG
  local config="$1"; shift
  WANT_DIGEST=1; DRY_RUN=0; OUTPUT_OVERRIDE=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --no-weight-digest) WANT_DIGEST=0; shift ;;
      --dry-run)          DRY_RUN=1; shift ;;
      --output-dir)       [[ $# -ge 2 ]] || die "--output-dir needs a value"; OUTPUT_OVERRIDE="$2"; shift 2 ;;
      *) die "unknown option '$1'" ;;
    esac
  done
  [[ -f "$config" ]] || die "no such config: $config"
  CONFIG_PATH="$(cd "$(dirname "$config")" && pwd)/$(basename "$config")"

  local tmp; tmp="$(mktemp -d)"; RESOLVED="$tmp/config-resolved.json"
  resolve_config "$CONFIG_PATH" "$RESOLVED"

  local exp_name base_model train_file dsm output_dir
  exp_name="$(cfg "$RESOLVED" experiment.name)"
  base_model="$(cfg "$RESOLVED" experiment.base_model)"
  train_file="$(cfg "$RESOLVED" data.train_file)"
  dsm="$(cfg "$RESOLVED" data.dataset_manifest)"
  output_dir="${OUTPUT_OVERRIDE:-$(cfg "$RESOLVED" runtime.output_dir "$HOME/experiments")}"

  [[ -f "$train_file" ]] || die "data.train_file does not exist: $train_file"
  assert_dataset_clean "$dsm"
  load_model_env "$base_model"
  [[ -d "$MODEL_DIR" ]] || die "base model weights not found: $MODEL_DIR (./modelctl download $base_model)"

  EXP_ID="$(new_experiment_id "$exp_name")"
  EXP_DIR="$output_dir/$EXP_ID"
  mkdir -p "$EXP_DIR"
  cp "$CONFIG_PATH" "$EXP_DIR/config.yaml"
  cp "$RESOLVED" "$EXP_DIR/config-resolved.json"

  local digest="-"
  if [[ "$WANT_DIGEST" == "1" ]]; then
    info "digesting base model weights (CONTRACTS §2.4) — this reads every file in $MODEL_DIR"
    digest="$(dir_digest "$MODEL_DIR")"
  else
    info "skipping the base-model weight digest — experiment will be marked nonconformant"
  fi

  export BASE_DIR HF_REPO HF_REVISION MODEL_DIR MODEL_ENV_FILE
  info "writing experiment manifest before training starts"
  write_manifest "$RESOLVED" "$CONFIG_PATH" "$EXP_DIR" "$EXP_ID" "$digest" >/dev/null
  TRAINER_CFG="$(write_trainer_config "$RESOLVED" "$EXP_DIR" "$MODEL_DIR")"
  info "experiment $EXP_ID prepared in $EXP_DIR"
}

cmd_config() {
  local tmp; tmp="$(mktemp -d)"
  resolve_config "$1" "$tmp/c.json"
  cat "$tmp/c.json"
  rm -rf "$tmp"
}

cmd_manifest() {
  prepare "$@"
  echo "$EXP_DIR/experiment-manifest.json"
}

cmd_run() {
  prepare "$@"
  local gpus entrypoint launcher arg_style extra visible
  gpus="$(cfg "$RESOLVED" runtime.gpus 8)"
  launcher="$(cfg "$RESOLVED" runtime.launcher torchrun)"
  entrypoint="$(cfg "$RESOLVED" runtime.entrypoint axolotl.cli.train)"
  arg_style="$(cfg "$RESOLVED" runtime.entrypoint_arg_style positional)"
  extra="$(cfg "$RESOLVED" runtime.extra_args "")"

  if command -v nvidia-smi >/dev/null 2>&1; then
    visible="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
    [[ "$visible" == "$gpus" ]] || info "WARNING: runtime.gpus=$gpus but $visible GPU(s) visible"
  fi

  local cmd=()
  case "$launcher" in
    torchrun)    cmd=("$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node="$gpus" -m "$entrypoint") ;;
    accelerate)  cmd=(accelerate launch --num_processes "$gpus" -m "$entrypoint") ;;
    python)      cmd=(python3 -m "$entrypoint") ;;
    *) die "unsupported runtime.launcher '$launcher' (torchrun|accelerate|python)" ;;
  esac
  case "$arg_style" in
    positional) cmd+=("$TRAINER_CFG") ;;
    flag)       cmd+=(--config "$TRAINER_CFG") ;;
    *) die "unsupported runtime.entrypoint_arg_style '$arg_style' (positional|flag)" ;;
  esac
  # shellcheck disable=SC2206
  [[ -n "$extra" ]] && cmd+=($extra)

  if [[ "$DRY_RUN" == "1" ]]; then
    info "dry run — would launch:"
    printf '%q ' "${cmd[@]}" >&2; echo >&2
    echo "$EXP_DIR"
    return 0
  fi

  local started; started="$(date +%s)"
  info "launching: ${cmd[*]}"
  set +e
  ( cd "$EXP_DIR" && "${cmd[@]}" ) 2>&1 | tee "$EXP_DIR/train.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "$rc" == "0" ]]; then
    finalize_manifest "$EXP_DIR" complete "$started"
    info "training complete — adapter in $EXP_DIR/adapter"
  else
    finalize_manifest "$EXP_DIR" failed "$started"
    info "training FAILED (exit $rc) — see $EXP_DIR/train.log"
  fi
  echo "$EXP_DIR"
  return "$rc"
}

cmd_doctor() {
  echo "repo         : $BASE_DIR"
  echo "weights dir  : $WEIGHTS_DIR"
  echo "holdout guard: $(python3 "$TRAINING_DIR/build_dataset.py" --print-holdout-constant)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  else
    echo "gpus         : nvidia-smi not found"
  fi
  local d
  for d in torch transformers peft trl accelerate bitsandbytes axolotl; do
    printf '%-14s: %s\n' "$d" "$(python3 -c "
try:
    import importlib.metadata as m
    print(m.version('$d'))
except Exception:
    print('not installed')
")"
  done
  command -v "$TORCHRUN_BIN" >/dev/null 2>&1 && echo "torchrun     : $(command -v "$TORCHRUN_BIN")" || echo "torchrun     : not found"
}

case "${1:-}" in
  doctor)   cmd_doctor ;;
  config)   [[ $# -ge 2 ]] || die "usage: ./training/train_lora.sh config <config.yaml>";   cmd_config "$2" ;;
  manifest) [[ $# -ge 2 ]] || die "usage: ./training/train_lora.sh manifest <config.yaml>"; shift; cmd_manifest "$@" ;;
  run)      [[ $# -ge 2 ]] || die "usage: ./training/train_lora.sh run <config.yaml>";      shift; cmd_run "$@" ;;
  -h|--help|help) sed -n '2,24p' "$0"; exit 0 ;;
  *) usage ;;
esac
