# nvidia-lambda — model serving for "The Harness Variable" (AI-P144)

One CLI, one OpenAI-compatible endpoint, six open models. The harness always talks to
`http://localhost:8000/v1`; which model answers is just `./modelctl switch <name>`.

## Quick start (on a Lambda instance)

```bash
pip install vllm hf_transfer
./modelctl list                     # see registry + what's on disk
./modelctl preflight qwen3-coder-next  # seconds: do the programs a launch execs resolve here?
./modelctl serve qwen3-coder-next   # downloads if needed, serves, waits for health
./modelctl test                     # one real completion through the endpoint
./modelctl switch kimi-k3           # stop current, serve next
./modelctl stop
```

## Registry

Each model is a file in `models.d/<name>.env` (HF repo, tensor-parallel size, extra
vLLM args). Add a model = add a file. Current lineup:

| name | GPUs | notes |
|---|---|---|
| qwen3-coder-next | 2× H100, FP8 (`models.d/qwen3-coder-next.env`) | NOT a 1× H100 model and not the cheap anchor the plan assumed — 79.7B BF16 params ≈ 159 GB against an 80 GB card. FP8-vs-BF16 is an open study-design call (AI-3158) |
| minimax-m3 | **unresolved — no H100 size in this repo can load it** | 427.0B params, BF16 only ≈ 854 GB: over 4× H100 (320 GB) *and* 8× H100 (640 GB). The env file's old "~230 GB FP8" was wrong — see TODO |
| deepseek-v4-flash | 8× H100 | use the retrained agentic checkpoint |
| glm-5.3 | 8× B200 | weights are out: `zai-org/GLM-5.3` is public, ungated, 141 safetensors files, `safetensors.total` 753,329,940,480 ≈ 755 GB — unchanged from the 8× B200 sizing already assumed. Phase 1 or Phase 3 reserve is a scheduling call now, not a blocker |
| kimi-k3 | 8× B200 (single node) | native MXFP4; `license:other` — its env file says accept the licence on HF first, and that note stands (see TODO) |
| qwen3.8-max | 2× 8× B200 (multi-node) | native FP8; needs Ray cluster; `license:other`, same licence note in its env file |

All parameter counts above are `safetensors.parameters` from the Hugging Face API, queried
2026-09-09Z (the timestamps in this file are UTC; the commit dates beside them are PDT).

Sizing lives with the model, not here: each `models.d/<name>.env` carries the arithmetic
(weights ÷ card, KV cache at `--max-model-len 262144`) behind its `TP` and `INSTANCE_TYPE`.
Where this table and an env file disagree, **the env file is right** — it is what launches
the instance, so a disagreement is a bug in one of the two and gets fixed rather than
narrated. One disagreement is still open and is listed under TODO: whether Lambda sells a
4× H100 type at all. `pricing/fallback-prices.json` lists `gpu_4x_h100_sxm5` at 1596¢/hr;
`models.d/qwen3-coder-next.env:25` says "Lambda has 1x, 2x and 8x H100 types; there is no
4x". Settling it needs a live `./lambdactl types` call, which needs `LAMBDA_API_KEY`.

The qwen3-coder-next row was wrong for the reason this paragraph exists: it read "80B/A3B"
as a small model, but A3B is 3B *active* parameters per token and all 79.7B must be
resident — ~159 GB at BF16 against an 80 GB card, and the FP8 sibling is ~80 GB, i.e. the
whole card before a single KV-cache block. Its **price is not settled here**: the only
figure in the repo is the comment on `models.d/qwen3-coder-next.env:33` ($8.38/hr for
`gpu_2x_h100_sxm5`), which nothing corroborates — `gpu_2x_h100_sxm5` has no entry in
`pricing/fallback-prices.json`, so a run on that instance type currently has **no fallback
price at all**. What is certain is only the negative: it is not the $2.49/hr
`gpu_1x_h100_pcie` the cost plan in RUNBOOK §5 was built on.

## Conventions (held constant across models — this is the study)

- `--max-model-len 262144` for every model
- weights live in `WEIGHTS_DIR` (defaults to `/persistent/models` if `/persistent`
  exists — use a Lambda persistent filesystem so weights survive instance restarts)
- pin the vLLM version in your run notes; Kimi K3 may need Moonshot's official
  docker image (uncomment `VLLM_DOCKER_IMAGE` in its env file)

## TODO before Day 1

Checked against the HF API 2026-09-09Z; every claim below is either verified or says
plainly that it is not.

- [ ] **minimax-m3 has no working configuration — this blocks Phase 1, not just the docs.**
      `MiniMaxAI/MiniMax-M3` is BF16 only: `safetensors.parameters` = `{BF16:
      426,993,800,960, F32: 46,339,200}`, so ~854 GB of weights against 320 GB (4× H100) or
      640 GB (8× H100). `MiniMaxAI/MiniMax-M3-FP8` is not pullable — the HF API answers an
      anonymous query with HTTP 401, i.e. it is absent or private.
      The official `MiniMaxAI/MiniMax-M3-MXFP8` is ~444 GB, which would need 8× H100, not
      the 4× the env file used to set. So this is a **checkpoint** decision (MXFP8, with the
      same quantisation confound as AI-3158, or B200), not a `TP` knob.
      `models.d/minimax-m3.env` is marked BLOCKED and now sets `TP=8` /
      `gpu_8x_h100_sxm5` so it cannot silently launch something too small — that is a
      not-too-small placeholder, **not** a configuration anyone has loaded.
- [ ] settle whether Lambda has a 4× H100 instance type (`./lambdactl types` with a live
      `LAMBDA_API_KEY`). `pricing/fallback-prices.json` lists one and
      `models.d/minimax-m3.env` used to select one; `models.d/qwen3-coder-next.env:25`
      says there is none. minimax-m3's sizing question cannot close while this is open.
- [ ] price `gpu_2x_h100_sxm5` from a live source and add it to
      `pricing/fallback-prices.json` — it is missing, so qwen3-coder-next has no fallback
      price, and the $8.38/hr in its env comment is unsourced.
- [ ] confirm `deepseek-ai/DeepSeek-V4-Flash` is the **retrained agentic** checkpoint. All
      six `HF_REPO` ids resolve (HTTP 200, none gated), so id verification is done — but
      *which* checkpoint sits behind that id is not something the API answers.
- [ ] verify Moonshot's vLLM docker image name for kimi-k3 (`VLLM_DOCKER_IMAGE`, still
      commented out in its env file)
- [ ] settle FP8 vs BF16 for qwen3-coder-next (AI-3158), then set its `TP` /
      `INSTANCE_TYPE` together with that decision
- [ ] dry-run the Ray two-node launch for qwen3.8-max (still not CI-supported)
- [ ] accept the Kimi K3 + Qwen3.8-Max licences on HF, as `models.d/kimi-k3.env:2` and
      `models.d/qwen3.8-max.env:2` say to. The anonymous API reports both repos
      `gated: false`, so there is probably no click-through left, but that is not the same
      evidence as "the account that pulls the weights can pull them" — so the env files'
      note stands and this item stays open. Both are `license:other` either way.

## GPU on / off — gpuctl

```bash
export LAMBDA_API_KEY=...
./gpuctl up kimi-k3 --serve      # launch the right instance for the model, lease it to you, start vLLM
./gpuctl status                  # what is alive, $/h, accrued, busy or idle
./gpuctl hold 4h                 # "still working on it"
./gpuctl ssh
./gpuctl down                    # off
```

**It turns itself off.** An instance stays alive only while it is *leased* (a job claimed it
with an expiry) or a harness process is running on it. `./gpuwatch` — run by CI every
15 minutes (`.github/workflows/reaper.yml`) — terminates anything else carrying this
project's `sh-` name prefix, plus hard caps (older than 24 h, or more than $500 accrued
across everything alive). `gpuctl up` leases for 2 h by default; a CI run leases for its
own 12 h ceiling. Forget a box and it costs you at most the lease, not the night.

The watchdog only ever touches `sh-*` names. Nothing in this repo can terminate an instance
that belongs to something else on the account.

## GPU lifecycle — lambdactl + CI (the low-level layer gpuctl sits on)

`lambdactl` drives the Lambda Cloud API (needs `LAMBDA_API_KEY`):

```bash
./lambdactl types                    # instance types + live availability ("<name>  $ 23.92/hr  <regions>")
./lambdactl up gpu_8x_b200_sxm6      # launch, wait for active + SSH, print "id ip"
./lambdactl ls
./lambdactl down <id|name>           # one instance; a name matching >1 instance is an ERROR, not a guess
./lambdactl down --all-named <name>  # every instance carrying that name tag (what CI teardown uses)
./lambdactl down --all               # kill everything
./lambdactl reap 24                  # kill anything older than 24h
```

CI (`.github/workflows/`):

- **benchmark.yml** — `workflow_dispatch(model, suite)`; the instance type derives from
  `models.d/<model>.env` `INSTANCE_TYPE`. Launch instance (name tag
  `ci-<model>-<run_id>-<run_attempt>`, unique per re-run attempt) → attach persistent weights
  filesystem → `modelctl serve` → run harness (3 passes) → `resultsctl package` each run
  directory named in the `RUN` lines on the instance → pull back **only**
  `run-manifest.json`, `SHA256SUMS`, `run-status.json` and the sealed bundle + `.sha256` into
  `results/<run_id>/` (never loose `trajectories/`, `patches/` or `results.jsonl` — §7.4) →
  **teardown in an `if: always()` step** that terminates by id AND by name tag and FAILS
  unless a follow-up `lambdactl ls` shows nothing live → cost ledger (`results/cost-log.jsonl`,
  priced from the run manifest's `price.effective_cents_per_hour`, falling back to the parsed
  list price; a zero is recorded as unknown). A package failure fails the job but never
  skips teardown. A `concurrency: gpu-run` group means only one GPU run at a time.
- **reaper.yml** — every 2h, terminates `ci-*` instances alive >14h and any instance alive
  >24h. The credit's seatbelt.
- **ci.yml** — on every push: shell syntax on every script (any failure fails the step),
  py_compile, adapter imports, the leakage-guard self-test, `resultsctl verify
  --git-hygiene --all` over the whole tracked tree, and the smoke test below.

Secrets: `LAMBDA_API_KEY`, `LAMBDA_SSH_PRIVATE_KEY`. Variables: `LAMBDA_SSH_KEY`, `LAMBDA_FS`,
`VLLM_VERSION` (pinned vLLM release installed on the instance).
Note: `harness/run.sh` is the interface CI expects — built under AI-2957.

## Smoke test — run this before spending a cent

```bash
./smoke/run-smoke.sh
```

Exercises the **entire** path — preflight, manifest, agent loop, tool calls, patch capture,
grading, aggregation, bundling, leakage guard — against a mock OpenAI-compatible endpoint
and a synthetic 2-task suite. No GPU, no credit, a few seconds.

It exists because every expensive bug found in review was a plumbing bug: a file never
shipped to the instance, a variable exported to the wrong step, two components disagreeing
about a field name. None are visible to a syntax check, and all of them would otherwise
surface on a $54/hr node after the weights had downloaded. Two real bugs were caught by its
first run — see `smoke/README.md`.

`.github/workflows/ci.yml` runs it on every push. Green CI = the harness is wired.

Requires `pytest` (the agenttask grader runs the hidden tests with it):
`python3 -m pip install pytest`.
