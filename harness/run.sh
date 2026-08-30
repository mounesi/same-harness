#!/usr/bin/env bash
# run.sh — run one benchmark suite against one served model, with full provenance.
# The harness is the CONTROL VARIABLE of AgentTask AI-P153: prompt, iteration budget,
# retry policy, sampling params and grading are identical for every model — only the
# weights change. Contract: docs/CONTRACTS.md §1 (this CLI) and §2 (run-manifest/v1).
#
#   ./harness/run.sh --model <name> --suite <name> [--passes N] [--out DIR] [options]
#
# Flags:
#   --model NAME        required. basename of a models.d/<name>.env. MUST equal the model
#                       the endpoint is actually serving, or the run refuses to start.
#   --suite NAME        required. swebench-verified | swebench-pro | agenttask | all
#                       'all' runs the three suites as three separate runs (own run id and
#                       manifest each, one shared run_group_id); exit code is the highest.
#   --passes N          independent passes per instance                    (default 3)
#   --out DIR           root of the output tree                            (default ./results)
#   --endpoint URL      OpenAI-compatible base url    ($HARNESS_ENDPOINT, else localhost:8000/v1)
#   --seed-file PATH    override the suite seed file (single suite only)
#   --partitions PATH   partition file                            (default suites/partitions.json)
#   --concurrency N     concurrent task attempts                           (default 4)
#   --task-timeout SEC  per-attempt wall-clock ceiling -> BUDGET_WALLCLOCK (default 1800)
#   --max-iters N       agent iteration budget. HELD CONSTANT — any other value marks the
#                       run flags.nonconformant                            (default 40)
#   --resume RUN_ID     continue an incomplete run in place: reuse its manifest and run only
#                       the missing (instance, pass) attempts
#   --limit N           debug only: first N instances in seed order (flags.exploratory)
#   --instance ID       debug only: run just this id; repeatable    (flags.exploratory)
#   --manifest-only     write the manifest(s), print the stdout line(s), exit 0
#   --dry-run           --manifest-only plus: probe the endpoint and render prompt-preview.txt
#   -h, --help          this text
#
# stdout is machine-readable ONLY — one line per run, printed when that run terminates:
#   RUN <run_id> <suite> <run_dir> <status>    status: complete | incomplete | failed |
#                                                      manifest-only | dry-run
# Everything human goes to stderr and is teed to <run_dir>/logs/harness.log.
#
# Exit codes (docs/CONTRACTS.md §1.3):
#   0 ok — tasks failing to resolve is a NORMAL 0     1 usage          2 config
#   3 preflight (endpoint down / served-model mismatch / unresolved REQUIRED provenance)
#   4 incomplete      5 grading degraded (>2% INFRA_GRADER)      130 interrupt
#
# Subprocess contract: this driver runs, once per pass and from the repo root,
#   python3 -m harness.agent run --manifest <m> --run-dir <d> --pass-idx <n>
#                                --summary-out <f> [--only-instances <f>]
# agent.py appends one raw-result/v1 record per attempt to <run_dir>/results.jsonl and
# MUST exit 0 iff it wrote a record for every planned attempt of that pass.
# The `-m` form is not cosmetic: `harness/types.py` (CONTRACTS.md §5.1) SHADOWS the
# stdlib `types` module when a script inside harness/ is executed by path, because
# Python puts the script's own directory first on sys.path. Every entry point in
# harness/ must be started as a module from the repo root, never as a file path.
#
# Env: HARNESS_ENDPOINT, WEIGHTS_DIR, STATE_DIR, HARNESS_PRICE_SNAPSHOT, HARNESS_PYTHON,
#      LAMBDA_INSTANCE_ID / LAMBDA_REGION / LAMBDA_INSTANCE_TYPE (exported by CI),
#      HARNESS_ALLOW_NETWORK=1 (allow the HfApi revision lookup),
#      HARNESS_SKIP_WEIGHT_DIGEST=1 (skip the weights content digest; marks nonconformant).
set -euo pipefail

ORIG_ARGV=("$@")

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$BASE_DIR/.." && pwd)"
MANIFEST_PY="$BASE_DIR/manifest.py"
AGENT_PY="$BASE_DIR/agent.py"
STATE_DIR="${STATE_DIR:-$REPO_DIR/.state}"
PY="${HARNESS_PYTHON:-python3}"

VALID_SUITES=(swebench-verified swebench-pro agenttask)

# Run a harness python entry point as a module from the repo root — see the header note
# on harness/types.py shadowing the stdlib. All paths handed to these MUST be absolute.
pyrun() {
  local mod="$1"; shift
  ( cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m "$mod" "$@" )
}
manifest_py() { pyrun harness.manifest "$@"; }

# ---------------------------------------------------------------- output helpers
LOGFILE=""
info() {
  printf '==> %s\n' "$*" >&2
  if [[ -n "$LOGFILE" && -f "$LOGFILE" ]]; then printf '==> %s\n' "$*" >>"$LOGFILE" || true; fi
}
usage()     { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0" >&2; }
usage_err() { printf 'error: %s\n\n' "$*" >&2; usage; exit 1; }
die_cfg()   { printf 'error: %s\n' "$*" >&2; exit 2; }
is_int()    { [[ "$1" =~ ^[0-9]+$ ]]; }
hex6()      { "$PY" -c 'import uuid; print(uuid.uuid4().hex[:6])'; }
utcstamp()  { date -u +%Y%m%dT%H%M%SZ; }
utciso()    { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ------------------------------------------------------------------ flag parsing
MODEL="" SUITE="" PASSES=3 OUT="./results"
ENDPOINT="${HARNESS_ENDPOINT:-http://localhost:8000/v1}"
SEED_FILE="" PARTITIONS="" CONCURRENCY=4 TASK_TIMEOUT=1800 MAX_ITERS=40
RESUME="" LIMIT="" MODE="exec"
INSTANCES=()
INVOCATION_FILE=""

need_arg() { [[ $# -ge 2 ]] || usage_err "$1 needs an argument"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        need_arg "$@"; MODEL="$2";        shift 2 ;;
    --suite)        need_arg "$@"; SUITE="$2";        shift 2 ;;
    --passes)       need_arg "$@"; PASSES="$2";       shift 2 ;;
    --out)          need_arg "$@"; OUT="$2";          shift 2 ;;
    --endpoint)     need_arg "$@"; ENDPOINT="$2";     shift 2 ;;
    --seed-file)    need_arg "$@"; SEED_FILE="$2";    shift 2 ;;
    --partitions)   need_arg "$@"; PARTITIONS="$2";   shift 2 ;;
    --concurrency)  need_arg "$@"; CONCURRENCY="$2";  shift 2 ;;
    --task-timeout) need_arg "$@"; TASK_TIMEOUT="$2"; shift 2 ;;
    --max-iters)    need_arg "$@"; MAX_ITERS="$2";    shift 2 ;;
    --resume)       need_arg "$@"; RESUME="$2";       shift 2 ;;
    --limit)        need_arg "$@"; LIMIT="$2";        shift 2 ;;
    --instance)     need_arg "$@"; INSTANCES+=("$2"); shift 2 ;;
    --manifest-only) if [[ "$MODE" != "dry-run" ]]; then MODE="manifest-only"; fi; shift ;;
    --dry-run)      MODE="dry-run"; shift ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; break ;;
    *)              usage_err "unknown flag: $1" ;;
  esac
done
[[ $# -eq 0 ]] || usage_err "unexpected argument: $1"

[[ -n "$MODEL" ]] || usage_err "--model is required"
[[ -n "$SUITE" ]] || usage_err "--suite is required"
is_int "$PASSES"       && [[ "$PASSES" -ge 1 ]]       || usage_err "--passes must be an integer >= 1"
is_int "$CONCURRENCY"  && [[ "$CONCURRENCY" -ge 1 ]]  || usage_err "--concurrency must be an integer >= 1"
is_int "$TASK_TIMEOUT" && [[ "$TASK_TIMEOUT" -ge 1 ]] || usage_err "--task-timeout must be an integer >= 1"
is_int "$MAX_ITERS"    && [[ "$MAX_ITERS" -ge 1 ]]    || usage_err "--max-iters must be an integer >= 1"
if [[ -n "$LIMIT" ]]; then
  is_int "$LIMIT" && [[ "$LIMIT" -ge 1 ]] || usage_err "--limit must be an integer >= 1"
fi
if [[ -n "$RESUME" ]]; then
  [[ "$SUITE" != "all" ]]      || usage_err "--resume names one run; it cannot be combined with --suite all"
  [[ -z "$LIMIT" ]]            || usage_err "--resume and --limit are mutually exclusive"
  [[ ${#INSTANCES[@]} -eq 0 ]] || usage_err "--resume and --instance are mutually exclusive"
  [[ "$MODE" == "exec" ]]      || usage_err "--resume cannot be combined with --manifest-only / --dry-run"
fi

# An unknown suite is a config value, not a flag typo -> exit 2 (CONTRACTS.md §1.3)
SUITE_LIST=()
if [[ "$SUITE" == "all" ]]; then
  SUITE_LIST=("${VALID_SUITES[@]}")
  [[ -z "$SEED_FILE" ]] || usage_err "--seed-file applies to a single suite, not --suite all"
else
  for s in "${VALID_SUITES[@]}"; do
    if [[ "$s" == "$SUITE" ]]; then SUITE_LIST=("$s"); fi
  done
  [[ ${#SUITE_LIST[@]} -eq 1 ]] || die_cfg "unknown suite '$SUITE' (want: ${VALID_SUITES[*]} | all)"
fi

MODEL_ENV_FILE="$REPO_DIR/models.d/$MODEL.env"
[[ -f "$MODEL_ENV_FILE" ]] || die_cfg "unknown model '$MODEL' — no $MODEL_ENV_FILE (see ./modelctl list)"
[[ -f "$MANIFEST_PY" ]] || die_cfg "missing $MANIFEST_PY"
if [[ "$MODE" == "exec" ]]; then
  [[ -f "$AGENT_PY" ]] || die_cfg "missing $AGENT_PY"
fi
[[ -n "$PARTITIONS" ]] || PARTITIONS="$REPO_DIR/suites/partitions.json"
[[ -f "$PARTITIONS" ]] || die_cfg "partitions file not found: $PARTITIONS"
abspath() { ( cd "$(dirname "$1")" 2>/dev/null && printf '%s/%s\n' "$(pwd)" "$(basename "$1")" ); }
PARTITIONS="$(abspath "$PARTITIONS")"
if [[ -n "$SEED_FILE" ]]; then
  [[ -f "$SEED_FILE" ]] || die_cfg "seed file not found: $SEED_FILE"
  SEED_FILE="$(abspath "$SEED_FILE")"
fi

default_seed_file() {
  case "$1" in
    swebench-verified) echo "$REPO_DIR/suites/verified-100.json" ;;
    swebench-pro)      echo "$REPO_DIR/suites/pro-50.json" ;;
    agenttask)         echo "$REPO_DIR/suites/agenttask/seed.json" ;;
  esac
}

# ------------------------------------------- model env, with modelctl's semantics
# Sourced exactly the way modelctl sources it (same defaults), so the manifest records
# what the server was actually launched with rather than a second-guessed parse.
load_model_env() {
  local kv line
  kv="$(
    set -euo pipefail
    # shellcheck disable=SC2034  # read back indirectly through ${!k} below
    HF_REPO="" HF_REVISION="" TP=1 PP=1 MAX_MODEL_LEN=262144
    # shellcheck disable=SC2034  # read back indirectly through ${!k} below
    EXTRA_ARGS="" MULTINODE=0 VLLM_DOCKER_IMAGE="" INSTANCE_TYPE=""
    # shellcheck source=/dev/null
    source "$MODEL_ENV_FILE"
    for k in HF_REPO HF_REVISION TP PP MAX_MODEL_LEN EXTRA_ARGS MULTINODE VLLM_DOCKER_IMAGE INSTANCE_TYPE; do
      printf 'HARNESS_MODELENV_%s=%s\n' "$k" "${!k}"
    done
  )" || die_cfg "could not source $MODEL_ENV_FILE"
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    export "${line%%=*}=${line#*=}"
  done <<<"$kv"
  [[ -n "${HARNESS_MODELENV_HF_REPO:-}" ]] || die_cfg "$MODEL_ENV_FILE does not define HF_REPO"
  export HARNESS_MODELENV_NAME="$MODEL"
}
load_model_env

if [[ -z "${WEIGHTS_DIR:-}" ]]; then
  if [[ -d /persistent ]]; then WEIGHTS_DIR=/persistent/models; else WEIGHTS_DIR="$HOME/models"; fi
fi
MODEL_DIR="$WEIGHTS_DIR/$MODEL"
case "$MODEL_DIR" in /*) ;; *) MODEL_DIR="$(pwd)/$MODEL_DIR" ;; esac

mkdir -p "$OUT" || die_cfg "cannot create --out directory: $OUT"
OUT_ABS="$(cd "$OUT" && pwd)"

# argv recorded verbatim in the manifest (NUL separated so nothing can be mangled)
INVOCATION_FILE="$(mktemp "${TMPDIR:-/tmp}/harness-invocation.XXXXXX")"
printf '%s\0' "$0" ${ORIG_ARGV[@]+"${ORIG_ARGV[@]}"} >"$INVOCATION_FILE"

RUN_GROUP_ID="${MODEL}___ALL___$(utcstamp)__$(hex6)"

# ------------------------------------------------------------- signal/exit traps
SIGNAL_EXIT=0
CUR_RUN_DIR="" CUR_RUN_ID="" CUR_SUITE="" CUR_STARTED="" CUR_FINALIZED=1

on_signal() {
  SIGNAL_EXIT="$1"
  info "signal received — no further passes will start; the run finalizes as incomplete"
}
trap 'on_signal 130' INT
trap 'on_signal 4'   TERM

cleanup() {
  local code=$?
  if [[ "$CUR_FINALIZED" == "0" && -n "$CUR_RUN_DIR" ]]; then
    info "abnormal exit ($code) — finalizing $CUR_RUN_ID as incomplete"
    finish_run incomplete 4 || true
  fi
  if [[ -n "$INVOCATION_FILE" ]]; then rm -f "$INVOCATION_FILE" || true; fi
}
trap cleanup EXIT

# --------------------------------------------------------------------- preflight
# Prints every model id the endpoint advertises, first one first. Non-zero if unreachable.
probe_endpoint() {
  local body
  body="$(curl -sf --max-time "${HARNESS_PROBE_TIMEOUT:-20}" "$ENDPOINT/models" 2>/dev/null)" || return 1
  printf '%s' "$body" | "$PY" -c '
import json, sys
try:
    data = json.load(sys.stdin).get("data") or []
except Exception:
    sys.exit(1)
print("\n".join(str(m.get("id", "")) for m in data))
' || return 1
}

# ----------------------------------------------------------------- finalisation
# finish_run <status> <exit_code> [extra finalize flags...]
# Rewrites the manifest, writes run-status.json and SHA256SUMS (last), prints the RUN line.
finish_run() {
  local status="$1" code="$2"; shift 2
  local SCAN_UNIQUE=0
  eval "$(manifest_py scan --run-dir "$CUR_RUN_DIR" --run-id "$CUR_RUN_ID" 2>/dev/null \
          || echo 'SCAN_UNIQUE=0')"
  manifest_py finalize --run-dir "$CUR_RUN_DIR" --status "$status" \
      --exit-code "$code" --attempts-written "$SCAN_UNIQUE" \
      --started-at "$CUR_STARTED" --ended-at "$(utciso)" "$@" >&2 \
    || info "WARNING: could not finalize the manifest for $CUR_RUN_ID"
  manifest_py checksums --run-dir "$CUR_RUN_DIR" >/dev/null 2>&1 \
    || info "WARNING: could not write SHA256SUMS for $CUR_RUN_ID"
  CUR_FINALIZED=1
  printf 'RUN %s %s %s %s\n' "$CUR_RUN_ID" "$CUR_SUITE" "$CUR_RUN_DIR" "$status"
}

# build_manifest <suite> <seed_file> <run_id> <run_dir> <passes> <served_model_name>
build_manifest() {
  local args i
  args=(
    build
    --repo "$REPO_DIR" --run-dir "$4" --run-id "$3" --run-group-id "$RUN_GROUP_ID"
    --model "$MODEL" --suite "$1" --seed-file "$2" --partitions "$PARTITIONS"
    --endpoint "$ENDPOINT" --weights-dir "$MODEL_DIR"
    --passes "$5" --concurrency "$CONCURRENCY" --task-timeout "$TASK_TIMEOUT"
    --max-iters "$MAX_ITERS" --served-model-name "$6"
    --invocation-file "$INVOCATION_FILE" --created-at "$CUR_STARTED"
    --mode "$MODE" --status running
  )
  if [[ -n "$LIMIT" ]]; then args+=(--limit "$LIMIT"); fi
  for i in ${INSTANCES[@]+"${INSTANCES[@]}"}; do args+=(--instance "$i"); done
  manifest_py ${args[@]+"${args[@]}"}
}

# ------------------------------------------------------------------------ do_run
# One suite, end to end. Prints exactly one RUN line unless it fails before the
# manifest exists (exit 1/2 => "nothing written"). Returns the run's exit code.
do_run() {
  local suite="$1" seed_file run_id run_dir rc=0 worst=0 p planned status code
  local passes="$PASSES" served="" ids="" mserved
  CUR_SUITE="$suite"

  seed_file="$SEED_FILE"
  [[ -n "$seed_file" ]] || seed_file="$(default_seed_file "$suite")"
  if [[ ! -f "$seed_file" ]]; then
    printf 'error: seed file not found: %s\n' "$seed_file" >&2
    return 2
  fi

  if [[ -n "$RESUME" ]]; then
    run_id="$RESUME"
    run_dir="$OUT_ABS/runs/$run_id"
    if [[ ! -f "$run_dir/run-manifest.json" ]]; then
      printf 'error: no run-manifest.json under %s — cannot --resume\n' "$run_dir" >&2
      return 2
    fi
    suite="$(manifest_py get "$run_dir/run-manifest.json" suite.name)"
    passes="$(manifest_py get "$run_dir/run-manifest.json" inference.passes)"
    CUR_SUITE="$suite"
    mkdir -p "$run_dir"/patches "$run_dir"/trajectories "$run_dir"/logs/attempts "$run_dir"/env
  else
    run_id="${MODEL}__${suite}__$(utcstamp)__$(hex6)"
    run_dir="$OUT_ABS/runs/$run_id"
    if [[ -e "$run_dir" ]]; then
      printf 'error: run directory already exists: %s\n' "$run_dir" >&2
      return 2
    fi
    mkdir -p "$run_dir"/patches "$run_dir"/trajectories "$run_dir"/logs/attempts "$run_dir"/env || return 2
  fi

  CUR_RUN_ID="$run_id"; CUR_RUN_DIR="$run_dir"; CUR_FINALIZED=0
  CUR_STARTED="$(utciso)"
  LOGFILE="$run_dir/logs/harness.log"
  : >>"$LOGFILE"

  info "run     : $run_id"
  info "model   : $MODEL  (${HARNESS_MODELENV_HF_REPO})"
  info "suite   : $suite   seed: $seed_file   passes: $passes"
  info "out     : $run_dir"
  info "mode    : $MODE"

  # ---- environment capture, before anything can perturb it ---------------
  cp "$MODEL_ENV_FILE" "$run_dir/env/model.env" 2>/dev/null || true
  "$PY" -m pip freeze --all 2>/dev/null | LC_ALL=C sort >"$run_dir/env/pip-freeze.txt" \
    || : >"$run_dir/env/pip-freeze.txt"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >"$run_dir/env/nvidia-smi.txt" 2>&1 || true
  else
    printf 'nvidia-smi unavailable on %s\n' "$(hostname 2>/dev/null || echo host)" \
      >"$run_dir/env/nvidia-smi.txt"
  fi
  if grep -h '^==> launching:' "$STATE_DIR/vllm.log" >/dev/null 2>&1; then
    grep -h '^==> launching:' "$STATE_DIR/vllm.log" | tail -n1 | sed 's/^==> launching: //' \
      >"$run_dir/env/vllm-args.txt"
  else
    printf 'unavailable: no "==> launching:" line in %s\n' "$STATE_DIR/vllm.log" \
      >"$run_dir/env/vllm-args.txt"
  fi

  # ---- preflight: the endpoint must be serving THIS model -----------------
  if [[ "$MODE" != "manifest-only" ]]; then
    info "preflight: GET $ENDPOINT/models"
    if ! ids="$(probe_endpoint)"; then
      cat >&2 <<EOF

  !!  ENDPOINT UNREACHABLE — REFUSING TO RUN  !!
      endpoint : $ENDPOINT
      model    : $MODEL
  GET \$endpoint/models did not answer, so nothing was executed.
  Bring the server up first:  ./modelctl serve $MODEL   (then ./modelctl status)

EOF
      if [[ -z "$RESUME" ]]; then
        build_manifest "$suite" "$seed_file" "$run_id" "$run_dir" "$passes" "" || true
      fi
      finish_run failed 3
      return 3
    fi
    served="$(printf '%s\n' "$ids" | head -n1)"
    if [[ "$served" != "$MODEL" ]]; then
      cat >&2 <<EOF

  !!  SERVED MODEL MISMATCH — REFUSING TO RUN  !!
      requested : $MODEL
      endpoint  : $ENDPOINT
      serving   : ${served:-<none>}
      advertised: $(printf '%s' "$ids" | tr '\n' ' ')
  Every record produced here would be attributed to the wrong model and would silently
  corrupt the whole study. Point --endpoint at the right server, or switch the model:
      ./modelctl switch $MODEL

EOF
      if [[ -z "$RESUME" ]]; then
        build_manifest "$suite" "$seed_file" "$run_id" "$run_dir" "$passes" "$served" || true
      fi
      finish_run failed 3
      return 3
    fi
    info "preflight: endpoint serves '$served' — matches --model"
  fi

  # ---- the manifest, BEFORE the first model call (codex invariant 1) ------
  if [[ -z "$RESUME" ]]; then
    rc=0
    build_manifest "$suite" "$seed_file" "$run_id" "$run_dir" "$passes" "$served" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      info "manifest build failed (exit $rc)"
      if [[ "$rc" -eq 3 ]]; then finish_run failed 3; return 3; fi
      # exit 2 is a config error: "nothing written". Take the half-built run dir with us
      # so nothing downstream ever sees a run directory without a manifest.
      CUR_FINALIZED=1
      if [[ ! -f "$run_dir/run-manifest.json" && "$run_dir" == "$OUT_ABS/runs/"* ]]; then
        rm -rf "$run_dir"
        LOGFILE=""
      fi
      return "$rc"
    fi
  else
    mserved="$(manifest_py get "$run_dir/run-manifest.json" model.served_model_name)"
    if [[ -n "$served" && "$mserved" != "$served" ]]; then
      printf 'error: this run was recorded against served model "%s" but the endpoint now serves "%s"\n' \
        "$mserved" "$served" >&2
      finish_run failed 3
      return 3
    fi
    info "resuming $run_id (manifest reused verbatim)"
  fi
  info "manifest: $run_dir/run-manifest.json"
  planned="$(manifest_py get "$run_dir/run-manifest.json" timing.attempts_planned)"

  if [[ "$MODE" == "manifest-only" ]]; then
    finish_run manifest-only 0
    return 0
  fi

  # ---- harness-constant invariant: every suite renders the SAME template --
  info "prompt  : asserting the $suite adapter renders the manifest's template id"
  local preview=()
  if [[ "$MODE" == "dry-run" ]]; then preview=(--write-preview "$run_dir/prompt-preview.txt"); fi
  if ! manifest_py prompt-check --repo "$REPO_DIR" \
        --manifest "$run_dir/run-manifest.json" ${preview[@]+"${preview[@]}"} >/dev/null; then
    finish_run failed 2
    return 2
  fi

  if [[ "$MODE" == "dry-run" ]]; then
    info "dry run : $planned attempts would be executed against $ENDPOINT — none were"
    finish_run dry-run 0
    return 0
  fi

  # ---- passes -------------------------------------------------------------
  for ((p = 0; p < passes; p++)); do
    if [[ "$SIGNAL_EXIT" != "0" ]]; then
      info "not starting pass $((p + 1)) — a termination signal was received"
      if [[ "$SIGNAL_EXIT" -gt "$worst" ]]; then worst="$SIGNAL_EXIT"; fi
      break
    fi
    local only=()
    if [[ -n "$RESUME" ]]; then
      local MISSING_COUNT=0
      eval "$(manifest_py missing --run-dir "$run_dir" --pass-idx "$p" \
                --out "$run_dir/logs/resume-pass-$p.ids")"
      if [[ "$MISSING_COUNT" -eq 0 ]]; then
        info "pass $((p + 1))/$passes — already complete, skipping"
        continue
      fi
      info "pass $((p + 1))/$passes — resuming $MISSING_COUNT missing attempt(s)"
      only=(--only-instances "$run_dir/logs/resume-pass-$p.ids")
    else
      info "pass $((p + 1))/$passes — $MODEL / $suite"
    fi

    set +e
    pyrun harness.agent run \
      --manifest "$run_dir/run-manifest.json" \
      --run-dir "$run_dir" \
      --pass-idx "$p" \
      --summary-out "$run_dir/logs/pass-$p.summary.json" \
      ${only[@]+"${only[@]}"} 2>&1 | tee -a "$LOGFILE" >&2
    rc="${PIPESTATUS[0]}"
    set -e

    if [[ "$rc" -ne 0 ]]; then
      info "pass $((p + 1)) exited $rc — not starting further passes"
      if [[ "$rc" -gt "$worst" ]]; then worst="$rc"; fi
      break
    fi
  done

  # ---- status -------------------------------------------------------------
  local SCAN_RECORDS=0 SCAN_UNIQUE=0 SCAN_RESOLVED=0 SCAN_INFRA_GRADER=0
  local SCAN_INFRA_UNKNOWN=0 SCAN_ATTEMPTS_SCORED=0 SCAN_MALFORMED=0 SCAN_FOREIGN=0
  eval "$(manifest_py scan --run-dir "$run_dir" --run-id "$run_id")"
  info "results : $SCAN_UNIQUE/$planned attempts written, $SCAN_RESOLVED resolved, \
$SCAN_ATTEMPTS_SCORED scored, $SCAN_INFRA_GRADER INFRA_GRADER, $SCAN_INFRA_UNKNOWN INFRA_UNKNOWN"
  if [[ "$SCAN_MALFORMED" -gt 0 ]]; then
    info "WARNING: $SCAN_MALFORMED unparseable line(s) in results.jsonl"
  fi
  if [[ "$SCAN_FOREIGN" -gt 0 ]]; then
    info "WARNING: $SCAN_FOREIGN record(s) in results.jsonl carry a foreign run_id"
  fi

  status="complete"; code=0
  local extra=()
  if [[ "$SIGNAL_EXIT" != "0" ]]; then
    status="incomplete"; code="$SIGNAL_EXIT"
  elif [[ "$worst" -ne 0 || "$SCAN_UNIQUE" -lt "$planned" ]]; then
    status="incomplete"; code=4
  fi
  if [[ "$SCAN_RECORDS" -gt 0 ]] && (( SCAN_INFRA_GRADER * 100 > SCAN_RECORDS * 2 )); then
    info "GRADING DEGRADED: $SCAN_INFRA_GRADER/$SCAN_RECORDS attempts ended INFRA_GRADER (>2%)"
    status="incomplete"; code=5; extra=(--grading-degraded)
  fi
  # A vLLM process that dies mid-run turns every remaining attempt into SERVER_*. Those count
  # in the denominator by design (CONTRACTS.md §4: an unservable model is a real cost of that
  # model) — which means a crashed server would otherwise publish as "this model scores ~0".
  # Distinguish infrastructure death from genuine unservability and refuse to call it complete.
  if [[ "$SCAN_RECORDS" -gt 0 ]] && (( SCAN_SERVER * 100 > SCAN_RECORDS * 10 )); then
    info "SERVER FAILURES: $SCAN_SERVER/$SCAN_RECORDS attempts ended SERVER_* (>10%),"
    info "                 $SCAN_TRAILING_SERVER of them consecutively at the end of the run."
    if (( SCAN_TRAILING_SERVER >= 5 )); then
      info "                 trailing burst => the endpoint died mid-run. NOT a model result."
    fi
    info "                 Do not publish this run: restart the server and --resume $run_id"
    status="incomplete"; code=4
  fi
  if [[ "$SCAN_RECORDS" -gt 0 ]] && (( SCAN_INFRA_SANDBOX * 100 > SCAN_RECORDS * 10 )); then
    info "SANDBOX FAILURES: $SCAN_INFRA_SANDBOX/$SCAN_RECORDS attempts ended INFRA_SANDBOX (>10%)"
    info "                  — workspaces are not being built; triage before publishing."
    status="incomplete"; code=4
  fi
  if [[ "$SCAN_RECORDS" -gt 0 ]] && (( SCAN_INFRA_UNKNOWN * 100 > SCAN_RECORDS * 2 )); then
    info "WARNING: INFRA_UNKNOWN is $SCAN_INFRA_UNKNOWN/$SCAN_RECORDS (>2%). Per CONTRACTS.md §4"
    info "         this run is INVALID: triage the exceptions and re-run before publication."
  fi
  if [[ "$status" == "incomplete" ]]; then
    info "re-runnable with: $0 --model $MODEL --suite $suite --out $OUT_ABS --resume $run_id"
  fi

  finish_run "$status" "$code" ${extra[@]+"${extra[@]}"}
  return "$code"
}

# ---------------------------------------------------------------------- dispatch
EXIT_CODE=0
for suite in "${SUITE_LIST[@]}"; do
  rc=0
  do_run "$suite" || rc=$?
  if [[ "$rc" -gt "$EXIT_CODE" ]]; then EXIT_CODE="$rc"; fi
  CUR_RUN_DIR=""; CUR_RUN_ID=""; CUR_FINALIZED=1; LOGFILE=""
done
exit "$EXIT_CODE"
