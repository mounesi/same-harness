#!/usr/bin/env bash
# run-smoke.sh — exercise the ENTIRE harness path end to end without a GPU.
#
# Why this exists: every expensive bug review found was a plumbing bug — a file that was
# never shipped to the instance, a variable exported to the wrong step, two components
# that disagreed about a field name. All of them are invisible to a syntax check and all
# of them would have surfaced on the first real run, after the weights download and the
# vLLM load, with the meter running at $54/hr. This runs the same path against a mock
# endpoint and a synthetic 2-task suite, in seconds, for free.
#
#   ./smoke/run-smoke.sh [--keep]
#
# It asserts, in order:
#   1. mock endpoint answers /v1/models         (run.sh preflight tier 1)
#   2. manifest builds and REQUIRED fields resolve                       (tier 2)
#   3. grading preflight passes for the suite                            (tier 3)
#   4. attempts execute and results.jsonl gets one record per attempt
#   5. every record carries the run_id, and every gradable record carries a grade block
#      stamped with the manifest's environment_digest (§2.3) — an ungraded record is a FAIL
#   6. the solved task is graded resolved=true, so the grader really ran
#   7. aggregate.py produces a headline table with a cost-per-resolved number
#   8. resultsctl packages a bundle whose checksums verify
#   9. the Phase-2 leakage guard fails closed on an unproven boundary, and — with the
#      boundary pinned — rejects the final_holdout attempt by name and builds a dataset
#      that holds the train task only
#
# Exit 0 = the pipeline is wired correctly. Anything else prints the failing stage.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SMOKE_PORT:-8099}"
MODEL="smoke-model"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

WORK="$(mktemp -d "${TMPDIR:-/tmp}/smoke-XXXXXX")"
PACK="$WORK/pack"
OUT="$WORK/results"
MOCK_PID=""

ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

cleanup() {
  [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true
  if [[ "$KEEP" == "1" ]]; then
    printf '\nkept: %s\n' "$WORK"
  else
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT

cd "$REPO"

step "0. synthetic suite"
python3 smoke/make_pack.py --out "$PACK" >/dev/null || fail "could not build the pack"
ok "2-task pack at $PACK"

step "1. mock endpoint"
MOCK_MODEL="$MODEL" python3 smoke/mock_endpoint.py --port "$PORT" --model "$MODEL" 2>"$WORK/mock.log" &
MOCK_PID=$!
for _ in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 0.25
done
curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL" || fail "mock endpoint never came up (see $WORK/mock.log)"
ok "serving $MODEL on :$PORT"

# The harness resolves the model from models.d/<name>.env; give it a smoke profile that
# points at the mock rather than at real weights.
cat > "models.d/$MODEL.env" <<EOF
# Synthetic profile used ONLY by smoke/run-smoke.sh. Not a study model.
HF_REPO="synthetic/smoke-model"
TP=1
MAX_MODEL_LEN=262144
EXTRA_ARGS=""
INSTANCE_TYPE="gpu_1x_h100_pcie"
EOF
trap 'rm -f "$REPO/models.d/'"$MODEL"'.env"; cleanup' EXIT

mkdir -p "$WORK/weights/$MODEL"
echo "synthetic" > "$WORK/weights/$MODEL/config.json"

step "2-4. run.sh: preflight, manifest, attempts"
set +e
AGENTTASK_DATA_DIR="$PACK" \
WEIGHTS_DIR="$WORK/weights" \
HARNESS_PARTITIONS="$PACK/partitions.json" \
HARNESS_SKIP_WEIGHT_DIGEST=1 \
./harness/run.sh \
  --model "$MODEL" --suite agenttask --passes 1 \
  --endpoint "http://127.0.0.1:$PORT/v1" \
  --seed-file "$PACK/seed.json" --partitions "$PACK/partitions.json" \
  --out "$OUT" --concurrency 2 --task-timeout 120 >"$WORK/run.out" 2>"$WORK/run.err"
RC=$?
set -e
cat "$WORK/run.err" | tail -25
[[ $RC -eq 0 ]] || fail "run.sh exited $RC (see $WORK/run.err)"
ok "run.sh exited 0"

RUN_LINE="$(grep '^RUN ' "$WORK/run.out" | head -1)" || fail "no RUN line on stdout"
RUN_ID="$(awk '{print $2}' <<<"$RUN_LINE")"
RUN_DIR="$(awk '{print $4}' <<<"$RUN_LINE")"
ok "run_id $RUN_ID"

[[ -f "$RUN_DIR/run-manifest.json" ]] || fail "no run-manifest.json"
[[ -f "$RUN_DIR/results.jsonl" ]]     || fail "no results.jsonl"
ok "manifest + results.jsonl written"

step "5-6. record integrity"
python3 - "$RUN_DIR" "$RUN_ID" <<'PY' || exit 1
import json, sys, pathlib
run_dir, run_id = pathlib.Path(sys.argv[1]), sys.argv[2]
man = json.loads((run_dir / "run-manifest.json").read_text())
recs = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
def die(m):
    print("  \033[31mFAIL\033[0m " + m); sys.exit(1)
if not recs:
    die("results.jsonl is empty — no attempt produced a record")
print("  \033[32mok\033[0m   %d record(s)" % len(recs))
for r in recs:
    if r.get("run_id") != run_id:
        die("record %s carries a foreign run_id" % r.get("attempt_id"))
print("  \033[32mok\033[0m   every record carries the run_id")
want = (man.get("harness") or {}).get("environment_digest")
if not want:
    die("manifest has no harness.environment_digest (CONTRACTS 2.3)")
# §2.3: every record that reached grading MUST carry grade.environment_digest equal to the
# manifest's. A missing grade block is only legitimate when the attempt never reached a
# gradable state (INFRA_*) or produced nothing to grade (NO_PATCH); anywhere else it is a
# FAIL — a None grade must never read as "matches".
def may_be_ungraded(code):
    return code.startswith("INFRA_") or code == "NO_PATCH"
bad = []
graded = 0
for r in recs:
    code = str(r.get("error_code") or "")
    grade = r.get("grade")
    if not isinstance(grade, dict):
        if not may_be_ungraded(code):
            bad.append("%s: no grade block (error_code %s)" % (r.get("attempt_id"), code or "?"))
        continue
    graded += 1
    got = grade.get("environment_digest")
    if got != want:
        bad.append("%s: grade.environment_digest %r != manifest %r" % (r.get("attempt_id"), got, want))
if bad:
    die("environment_digest check failed (CONTRACTS 2.3):\n         " + "\n         ".join(bad))
if graded == 0:
    die("no record carries a grade block — nothing was graded")
print("  \033[32mok\033[0m   %d/%d records graded, every grade carries the manifest's environment_digest (2.3)"
      % (graded, len(recs)))
resolved = [r for r in recs if r.get("resolved") is True]
codes = sorted({r.get("error_code") for r in recs})
print("  \033[32mok\033[0m   error codes: %s" % codes)
if not resolved:
    die("nothing resolved — the grader never confirmed a real fix (codes: %s)" % codes)
print("  \033[32mok\033[0m   %d attempt(s) graded resolved=true — the grader really ran" % len(resolved))
PY

step "7. aggregate"
set +e
# --include-nonconformant is REQUIRED here and that is the point: this run deliberately
# deviates (120s task timeout, skipped weight digest, dirty tree), so the flag split
# correctly excludes it from a headline table. Being forced to pass the flag is the
# aggregator proving it will not silently publish a non-study run.
python3 analysis/aggregate.py --manifests "$RUN_DIR/run-manifest.json" \
  --include-nonconformant \
  --out "$WORK/agg" >"$WORK/agg.out" 2>"$WORK/agg.err"
AGG=$?
set -e
[[ $AGG -eq 0 ]] || { tail -20 "$WORK/agg.err"; fail "aggregate.py exited $AGG"; }
grep -qi "resolve" "$WORK/agg.out" || { head -40 "$WORK/agg.out"; fail "no headline table"; }
ok "headline table produced"
sed -n '1,18p' "$WORK/agg.out" | sed 's/^/      /'

step "8. bundle"
set +e
./resultsctl package "$RUN_DIR" --dist "$WORK/dist" >"$WORK/pkg.out" 2>&1
PKG=$?
set -e
if [[ $PKG -eq 0 ]]; then
  ok "packaged"
  ./resultsctl verify "$RUN_DIR" >/dev/null 2>&1 && ok "checksums verify" || fail "checksum verify failed"
else
  tail -10 "$WORK/pkg.out"; fail "resultsctl package exited $PKG"
fi

step "9. leakage guard"
# Three things must hold, and they are different:
#   (a) the guard FAILS CLOSED when the holdout boundary is not provable — the committed
#       constant is UNFROZEN, or frozen against a partitions.json that is not this pack's;
#   (b) with the boundary pinned, it REJECTS a run containing a holdout attempt, BY NAME;
#   (c) with the boundary pinned, a run of train-only attempts builds a dataset that holds
#       smoke-0001 and not smoke-0002.
# (b) and (c) run against a COPY of the guard pinned to this pack's holdout checksum, so they
# exercise the real code paths whatever the repo's constant says.
set +e
python3 training/build_dataset.py --manifests "$RUN_DIR/run-manifest.json" \
  --partitions "$PACK/partitions.json" --split train --dry-run \
  >"$WORK/ds.out" 2>&1
DS=$?
set -e
if [[ $DS -eq 0 ]]; then
  # This run CONTAINS the final_holdout attempt (smoke-0002); guard 3 must refuse it, so a
  # zero exit means the guard let a holdout attempt through.
  tail -6 "$WORK/ds.out"; fail "guard accepted a run containing final_holdout attempt smoke-0002 (exit 0)"
elif grep -qi "not pinned\|FINAL_HOLDOUT_SHA256\|modified since freeze" "$WORK/ds.out"; then
  ok "(a) fails closed when the holdout boundary is not provable (exit $DS)"
elif grep -q "agenttask::smoke-0002" "$WORK/ds.out"; then
  ok "(a) boundary provable in-tree: rejects the final_holdout attempt by name (exit $DS)"
else
  tail -12 "$WORK/ds.out"; fail "guard failed for an unexpected reason (exit $DS)"
fi

# pin a copy of the guard to THIS pack's holdout checksum (the value --freeze would compile in)
HOLDOUT_SHA="$(python3 - "$PACK/partitions.json" <<'PY'
import json, sys
sys.path.insert(0, "training")
import build_dataset as bd
parts = json.load(open(sys.argv[1]))
print(bd.id_list_sha256(sorted(bd.holdout_ids(parts))))
PY
)" || fail "could not compute the pack's holdout checksum"
[[ "$HOLDOUT_SHA" =~ ^[0-9a-f]{64}$ ]] || fail "holdout checksum is not hex64: '$HOLDOUT_SHA'"
mkdir -p "$WORK/guard"
sed "s/^FINAL_HOLDOUT_SHA256 = \".*\"\$/FINAL_HOLDOUT_SHA256 = \"$HOLDOUT_SHA\"/" \
  training/build_dataset.py >"$WORK/guard/build_dataset.py"
grep -q "^FINAL_HOLDOUT_SHA256 = \"$HOLDOUT_SHA\"" "$WORK/guard/build_dataset.py" \
  || fail "could not pin a guard copy to the pack holdout"
GUARD="$WORK/guard/build_dataset.py"

# (b) the run as produced holds the holdout attempt -> refused, and smoke-0002 is NAMED
set +e
python3 "$GUARD" --manifests "$RUN_DIR/run-manifest.json" \
  --partitions "$PACK/partitions.json" --split train --dry-run >"$WORK/ds-reject.out" 2>&1
DR=$?
set -e
[[ $DR -ne 0 ]] || { tail -6 "$WORK/ds-reject.out"; fail "pinned guard accepted a run containing the final_holdout attempt"; }
grep -q "agenttask::smoke-0002" "$WORK/ds-reject.out" \
  || { tail -8 "$WORK/ds-reject.out"; fail "pinned guard refused the run but did not name smoke-0002 (exit $DR)"; }
ok "(b) boundary pinned: rejects the final_holdout attempt by name (exit $DR)"

# (c) a train-only copy of the run (holdout record removed) -> exactly the train task is kept
SRC="$WORK/ds-src/$RUN_ID"
mkdir -p "$WORK/ds-src" && cp -R "$RUN_DIR" "$SRC"
python3 - "$SRC/results.jsonl" <<'PY' || fail "could not derive the train-only results copy"
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
keep = [l for l in p.read_text().splitlines()
        if l.strip() and json.loads(l).get("instance_id") != "smoke-0002"]
assert keep, "no train records left"
p.write_text("\n".join(keep) + "\n")
PY
python3 "$GUARD" --manifests "$SRC/run-manifest.json" --partitions "$PACK/partitions.json" \
  --split train --dry-run >"$WORK/ds-dry.out" 2>"$WORK/ds-dry.err" \
  || { tail -8 "$WORK/ds-dry.err"; fail "pinned guard dry-run failed on a train-only run"; }
python3 - "$WORK/ds-dry.out" <<'PY' || fail "dry-run stats do not show exactly the train task kept"
import json, sys
last = [l for l in open(sys.argv[1]).read().splitlines() if l.strip()][-1]
d = json.loads(last)
st = d["stats"]
assert d["split"] == "train", d
assert st["considered"] == 1 and st["kept"] == 1, st
print("  \033[32mok\033[0m   (c) dry-run stats: considered=%d kept=%d" % (st["considered"], st["kept"]))
PY
python3 "$GUARD" --manifests "$SRC/run-manifest.json" --partitions "$PACK/partitions.json" \
  --split train --out "$WORK/ds" >"$WORK/ds-write.out" 2>"$WORK/ds-write.err" \
  || { tail -8 "$WORK/ds-write.err"; fail "pinned guard could not write the train-only dataset"; }
DATA="$(awk 'NF{p=$1} END{print p}' "$WORK/ds-write.out")"
[[ -n "$DATA" && -f "$DATA" ]] || fail "build_dataset.py printed no dataset path (got: '$DATA')"
grep -q "smoke-0001" "$DATA" || fail "train task smoke-0001 missing from the written dataset $DATA"
if grep -q "smoke-0002" "$DATA"; then fail "final_holdout task smoke-0002 leaked into the written dataset $DATA"; fi
ok "(c) written dataset holds smoke-0001 and not smoke-0002 ($(basename "$DATA"))"

# and the rejection path itself, which is unit-tested
set +e
python3 training/build_dataset.py --self-test >"$WORK/guard.out" 2>&1
GT=$?
set -e
[[ $GT -eq 0 ]] || { tail -12 "$WORK/guard.out"; fail "leakage-guard self-test failed"; }
ok "leakage-guard self-test: $(grep -o 'Ran [0-9]* tests' "$WORK/guard.out" | head -1) passed"

printf '\n\033[32m=== SMOKE PASSED ===\033[0m the pipeline is wired end to end.\n'
