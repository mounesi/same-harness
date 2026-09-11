#!/usr/bin/env bash
# demo.sh — one command: bring a GPU up, have a model build the racing-v1 game, bring it down.
#
#   ./suites/gamespec/demo.sh [model] [--hold 2h] [--passes 1] [--keep-up]
#
#   model     a models.d/<name>.env basename. Default: shakedown-qwen30b — the cheapest
#             Qwen coder that is KNOWN to serve on 1x H100 ($3.29/h). qwen3-coder-next as
#             configured (BF16, 2x H100) cannot load; see its env file before choosing it.
#   --hold    lease length handed to gpuctl (default 2h). The box is torn down at the end
#             of this script regardless; the lease only covers a crash of this script.
#   --keep-up leave the instance running afterwards (you then own `./gpuctl down`).
#
# What it does, in order:
#   1. ./gpuctl up <model> --serve --hold <ttl>       launch, lease, ship repo, start vLLM
#   2. ssh: ./harness/run.sh --model <model> --suite gamespec --passes N --out ~/results
#   3. ssh: ./resultsctl package <run_dir>            sealed bundle (manifest, results, checksums)
#   4. scp the run's patches/ + results.jsonl back to results/gamespec/<run_id>/
#   5. rebuild game.html locally from the pass-0 diff and run the floor check on it
#   6. ./gpuctl down --yes                            retrieves the sealed bundle, then terminates
#
# Env (same as gpuctl): LAMBDA_API_KEY (required), LAMBDA_FS, VLLM_VERSION, LAMBDA_IMAGE,
# LAMBDA_SSH_KEY. The three repo variables CI uses are filled in as defaults below when
# unset, so a laptop that only exports the API key can run this.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_USER="${SSH_USER:-ubuntu}"
VENV_NAME="${VENV_NAME:-harness-venv}"
die()  { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*" >&2; }

MODEL="shakedown-qwen30b"; HOLD="2h"; PASSES=1; KEEP_UP=0
while [[ $# -gt 0 ]]; do case "$1" in
  --hold)    HOLD="$2"; shift 2 ;;
  --passes)  PASSES="$2"; shift 2 ;;
  --keep-up) KEEP_UP=1; shift ;;
  -h|--help) sed -n '2,25p' "$0" >&2; exit 0 ;;
  -*) die "unknown flag $1" ;;
  *) MODEL="$1"; shift ;;
esac; done

[[ -n "${LAMBDA_API_KEY:-}" ]] || die "LAMBDA_API_KEY not set (https://cloud.lambdalabs.com/api-keys)"
[[ -f "$HERE/models.d/$MODEL.env" ]] || die "no models.d/$MODEL.env"
# Defaults mirror the repo variables benchmark.yml runs with (gh variable list).
export LAMBDA_FS="${LAMBDA_FS:-harness-weights-usw3}"
export VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
export LAMBDA_IMAGE="${LAMBDA_IMAGE:-gpu-base-24-04:24.4.4-2141}"
export LAMBDA_SSH_KEY="${LAMBDA_SSH_KEY:-harness-macbook}"
command -v node >/dev/null || die "node is needed locally to floor-check the result (brew install node)"

cd "$HERE"

# ---- 1. up -----------------------------------------------------------------------------
info "bringing up $MODEL (lease $HOLD)"
UP="$("$HERE/gpuctl" up "$MODEL" --serve --hold "$HOLD" | tail -n1)"
IP="$(awk '{print $2}' <<<"$UP")"
[[ -n "$IP" ]] || die "gpuctl up printed no IP: $UP"
info "instance is up at $IP and serving $MODEL"
# The harness must be told where the weights are, the same way benchmark.yml tells it:
# run.sh defaults WEIGHTS_DIR to /persistent/models, else ~/models, and the persistent
# filesystem is mounted at neither. Without this the manifest cannot resolve
# model.weight_digest (REQUIRED) and refuses to start — the first demo lost ~10 min of
# H100 to exactly that. Resolved from the API like gpuctl does, not guessed.
MOUNT="$("$HERE/lambdactl" fs "$LAMBDA_FS" | awk '{print $3}')"
[[ -n "$MOUNT" && "$MOUNT" != "-" ]] || die "filesystem '$LAMBDA_FS' reports no mount point"
WEIGHTS_DIR="${WEIGHTS_DIR_OVERRIDE:-$MOUNT/models}"
info "weights dir for the harness: $WEIGHTS_DIR"

teardown() {
  if (( KEEP_UP )); then
    info "--keep-up: leaving the instance running. It is leased for $HOLD; ./gpuctl down when done."
    return
  fi
  info "tearing down (gpuctl down retrieves any sealed bundle first)"
  "$HERE/gpuctl" down --yes || echo "WARNING: gpuctl down failed — run ./gpuctl status and ./gpuctl down by hand" >&2
}
trap teardown EXIT

sshto() { ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$SSH_USER@$IP" "$@"; }

# ---- 1b. node — the one grading dependency the Lambda image does NOT carry -----------
# suites/gamespec/floor_check.py drives SimCore headlessly in node, and grading preflight
# (tier 3) refuses to start a run on a host without it — which is exactly what the second
# demo hit, after vLLM was already serving (~17 min of H100). A pinned LTS tarball goes into
# the venv's own bin/, the directory the harness step already puts first on PATH, so no
# sudo, no apt, and the version is the same on every box.
NODE_VERSION="${NODE_VERSION:-22.12.0}"
NODE_SHA256="${NODE_SHA256:-22982235e1b71fa8850f82edd09cdae7e3f32df1764a9ec298c72d25ef2c164f}"
info "ensuring node $NODE_VERSION is in ~/$VENV_NAME/bin"
sshto "set -e
if ~/$VENV_NAME/bin/node --version 2>/dev/null | grep -qx 'v$NODE_VERSION'; then echo 'node already present'; exit 0; fi
cd /tmp && curl -fsSLO 'https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz'
echo '$NODE_SHA256  node-v$NODE_VERSION-linux-x64.tar.xz' | sha256sum -c - >/dev/null
tar -xJf node-v$NODE_VERSION-linux-x64.tar.xz -C ~/$VENV_NAME --strip-components=1 --exclude='*/share' --exclude='*/include' --exclude='CHANGELOG.md' --exclude='README.md' --exclude='LICENSE'
rm -f node-v$NODE_VERSION-linux-x64.tar.xz
~/$VENV_NAME/bin/node --version" || die "could not install node into the venv on the instance"

# ---- 2. run the suite -----------------------------------------------------------------
# Same PATH/HARNESS_PYTHON pair RUNBOOK §1.4 and benchmark.yml export: the venv is hermetic
# and nothing is in the image's python3. node is installed into
# the same venv bin/ above; grading preflight (tier 3) refuses to start if it is missing.
# The first run also hashes the weights for model.weight_digest (31 GB for the default
# model, a few minutes); the digest is cached next to the weights, so later runs skip it.
info "running gamespec ($PASSES pass(es)) against the served model"
RUN_LINE="$(sshto "cd ~/harness-repo && export PATH=\"\$HOME/$VENV_NAME/bin:\$PATH\" HARNESS_PYTHON=\"\$HOME/$VENV_NAME/bin/python\" WEIGHTS_DIR='$WEIGHTS_DIR' \
  && ./harness/run.sh --model '$MODEL' --suite gamespec --passes '$PASSES' --out ~/results" | grep '^RUN ' | tail -n1)" \
  || die "run.sh failed on the instance (the box is still up until teardown; ./gpuctl ssh to inspect ~/results)"
RUN_ID="$(awk '{print $2}' <<<"$RUN_LINE")"
RUN_DIR="$(awk '{print $4}' <<<"$RUN_LINE")"
STATUS="$(awk '{print $5}' <<<"$RUN_LINE")"
[[ -n "$RUN_ID" ]] || die "no RUN line from run.sh"
info "run $RUN_ID finished: $STATUS"

# ---- 3. package (the sealed bundle gpuctl down will retrieve) -------------------------
sshto "cd ~/harness-repo && export PATH=\"\$HOME/$VENV_NAME/bin:\$PATH\" && ./resultsctl package '$RUN_DIR' --dist ~/results/dist" \
  || echo "WARNING: resultsctl package failed; the raw run dir is still pulled below" >&2

# ---- 4. pull the artifact we actually came for ---------------------------------------
# A sealed bundle deliberately EXCLUDES patches/ and trajectories/ (CONTRACTS §7.4). The
# game lives in the patch and the model's reasoning in the trajectory, so copy both by
# hand, before teardown — the first live run pulled only patches, and the one question
# worth asking afterwards ("what did it do for 26 iterations?") had left with the box.
# gamespec is CONSENT_CLASS public, so trajectories may sit in results/ (git-ignored).
LOCAL="$HERE/results/gamespec/$RUN_ID"
mkdir -p "$LOCAL"
scp -q -r -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  "$SSH_USER@$IP:$RUN_DIR/patches" "$SSH_USER@$IP:$RUN_DIR/trajectories" "$SSH_USER@$IP:$RUN_DIR/logs" \
  "$SSH_USER@$IP:$RUN_DIR/results.jsonl" "$SSH_USER@$IP:$RUN_DIR/run-manifest.json" "$LOCAL/" \
  || die "could not copy the run back — the box is still up until teardown; ./gpuctl ssh and copy $RUN_DIR by hand"
info "pulled patches + results to $LOCAL"

# ---- 5. rebuild game.html from the diff and floor-check it locally --------------------
python3 - "$LOCAL" <<'PY'
import json, sys, pathlib
local = pathlib.Path(sys.argv[1])
recs = [json.loads(l) for l in (local / "results.jsonl").read_text().splitlines() if l.strip()]
for r in recs:
    print("   %-12s pass %s  resolved=%s  %s  floor %s" % (
        r.get("instance_id"), r.get("pass_idx"), r.get("resolved"), r.get("error_code"),
        (r.get("grade") or {}).get("fail_to_pass")))
PY
# The diff is against the adapter's base tree (SPEC.md, floor_check.py, README, .gitignore),
# not an empty directory, so rebuild through the adapter — it lays the base down and applies
# the diff exactly the way grade() does. Needs a Python >= 3.11 locally, like the harness.
REBUILD_PY="${HARNESS_PYTHON:-python3}"
for diff in "$LOCAL"/patches/*/pass-*.diff; do
  [[ -f "$diff" ]] || continue
  iid="$(basename "$(dirname "$diff")")"; pass="$(basename "$diff" .diff)"
  out="$LOCAL/built/$iid/$pass"; rm -rf "$out"
  if game="$(cd "$HERE" && "$REBUILD_PY" -m harness.adapters.gamespec rebuild "$iid" "$diff" "$out" 2>"$LOCAL/rebuild-$iid-$pass.err")"; then
    info "$iid $pass: rebuilt -> $game"
    spec_flag=(); [[ "$iid" == "racing-v1" ]] || spec_flag=(--spec "$iid")
    "$REBUILD_PY" "$HERE/suites/gamespec/floor_check.py" "$game" "${spec_flag[@]}" || true
  else
    info "$iid $pass: could not rebuild the deliverable: $(tail -1 "$LOCAL/rebuild-$iid-$pass.err")"
  fi
done

info "done. Open results/gamespec/$RUN_ID/built/racing-v1/pass-0/game.html in a browser to play it."
