# Runbook — The Harness Variable

Operating guide for running the benchmark end to end. The contract every number below
rests on is `docs/CONTRACTS.md`; this file is the "how", that one is the "what counts".

The one-sentence design: **the harness is the control variable.** Prompt, iteration budget,
retry policy, sampling parameters, context length and grading are byte-identical for every
model; only the weights change. Anything that varies per model shows up in the run manifest
as `flags.nonconformant` and is excluded from the headline table by default.

---

## 1. How to run it

### 1.1 Prove the pipeline first (free, ~10 s, no GPU)

```bash
python3 -m pip install pytest          # the agenttask grader runs hidden tests with it
./smoke/run-smoke.sh
```

Runs the entire path — preflight, manifest, agent loop, tool calls, patch capture, grading,
aggregation, bundling, leakage guard — against a mock endpoint and a synthetic 2-task suite.
If this is red, do not spend a cent. CI runs it on every push (`.github/workflows/ci.yml`).

### 1.2 Pre-flight — once, before any paid run (blocking)

These are one-time and every one of them blocks the first dispatch (the reaper needs no setup — it is on, and skips cleanly until the secrets exist):

| # | Step | Command / where |
|---|---|---|
| 1 | Confirm dataset access, then generate the real seed files | `python3 suites/generate_seeds.py` → `suites/verified-100.json`, `suites/pro-50.json` |
| 2 | **Freeze partitions, then pin the holdout hash** | `python3 suites/generate_partitions.py` then `python3 training/build_dataset.py --freeze suites/partitions.json --write` |
| 3 | Add GitHub secrets + variables | see §2.2 — `benchmark.yml` refuses to launch without them |
| 4 | Create the Lambda persistent filesystem for weights | Lambda console; its name is `LAMBDA_FS` |
| 5 | Accept the Kimi K3 (and Qwen3.8-Max) license on Hugging Face | from the org account, or `hf download` fails at instance time |
| 6 | Pin the vLLM version | set `VLLM_VERSION`; on the first instance re-resolve `harness/requirements.lock` |

The harness enforces (2): `manifest.py build` refuses a `"placeholder": true` partitions
file, and `benchmark.yml` checks it **before** launching an instance. Freezing partitions
*after* Phase 1 invalidates Phase 2 (the leakage guard is keyed to the frozen hash).

### 1.3 A benchmark run — the normal path (CI)

Dispatch `benchmark-run` from the Actions tab (or `gh workflow run benchmark.yml`) with:

- `model` — one of `qwen3-coder-next | minimax-m3 | deepseek-v4-flash | glm-5.3 | kimi-k3`
  (the instance type derives from `models.d/<model>.env`, never chosen by hand)
- `suite` — `swebench-verified | swebench-pro | agenttask` (**one suite per dispatch**;
  `all` runs three suites sequentially and will not fit the 12 h job ceiling on a large model)

What the workflow does, in order: launch instance (`lambdactl up`) → ship the repo →
install pinned deps → `modelctl serve` → `harness/run.sh` (3 passes) → package each run
(`resultsctl package`) → pull back **only** manifests, checksums and sealed bundles → write
the cost ledger → **tear the instance down in an `if: always()` step**. Only one GPU job
runs at a time (`concurrency: gpu-run`).

Run **`qwen3-coder-next` on `swebench-verified` first.** It is the cheapest model (~$2.49/hr)
and exists to shake down the real path — weights download, vLLM at TP=1, docker grading —
before anything expensive.

### 1.4 A benchmark run — by hand (on a Lambda instance)

```bash
# your machine — one command up, one command down
export LAMBDA_API_KEY=... LAMBDA_FS=...
./gpuctl up kimi-k3 --serve --hold 6h          # launches the model's instance type, leases it to
                                               # you for 6 h, ships the repo, starts vLLM
./gpuctl ssh

# on the instance
./harness/run.sh --model kimi-k3 --suite swebench-verified --passes 3 --out ~/results
./resultsctl package ~/results/runs/<run_id>    # run_id is field 2 of the RUN line run.sh prints
./resultsctl upload dist/<run_id>.tar.gz && ./resultsctl index dist/<run_id>.tar.gz

# your machine
./gpuctl down
```

**The instance turns itself off if you forget.** It lives only while leased or while a harness
process is running; `./gpuwatch` (CI, every 15 min) terminates it otherwise. Running long?
`./gpuctl hold 4h`. Want to see the meter? `./gpuctl status`. The low-level tools
(`lambdactl`, `modelctl`) are still there underneath.

`run.sh` prints exactly one machine-readable line per run on stdout —
`RUN <run_id> <suite> <run_dir> <status>` — and everything human on stderr.

**Exit codes** (`docs/CONTRACTS.md §1.3`): `0` ok (tasks failing to resolve is a normal 0) ·
`1` usage · `2` config · `3` preflight (endpoint down, served model ≠ `--model`, grading
dependency missing, digest mismatch) · `4` incomplete · `5` grading degraded · `130` interrupt.
Exit 0 is only ever reported when the manifest finalized **and** `SHA256SUMS` was written.

**Resume** an interrupted run in place: `./harness/run.sh --model M --suite S --resume <run_id>`.
Attempts that ended `INFRA_HOST` (the host went away) are re-run, their stale records
pruned first, so every attempt keeps exactly one record. A resume that fails preflight
leaves the run directory untouched.

### 1.5 Analysis, after the runs

```bash
python3 analysis/aggregate.py --manifests results/*/run-manifest.json \
    --cost-log results/cost-log.jsonl --api-pricing analysis/api-pricing.json --out-dir report/
```

It takes an **explicit list of manifests** (directories are refused by design), verifies
checksums, refuses to mix runs whose harness constants differ (`--allow-mixed` to override,
loudly), excludes `flags.nonconformant` runs by default, and includes
`flags.provenance_incomplete` runs with their cost columns marked approximate.

---

## 2. What variables to set

### 2.1 Per model — `models.d/<model>.env` (committed)

| key | meaning | held constant? |
|---|---|---|
| `HF_REPO` | Hugging Face repo id of the weights | per model |
| `TP` / `PP` | tensor / pipeline parallel size | per model |
| `INSTANCE_TYPE` | Lambda instance type CI launches | per model |
| `EXTRA_ARGS` | extra vLLM flags (e.g. `--trust-remote-code`) | per model — **cannot** override a held constant: `modelctl` appends the study values last, and repeating `--max-model-len`/`--served-model-name` here marks the run nonconformant |
| `MAX_MODEL_LEN` | context length | **must be `262144`** — any other value → `nonconformant` |
| `MULTINODE` | `1` for multi-node models (Qwen3.8-Max) | not CI-supported yet |

### 2.2 CI — GitHub repository settings

| kind | name | value |
|---|---|---|
| secret | `LAMBDA_API_KEY` | Lambda Cloud API key (`cloud.lambdalabs.com/api-keys`) |
| secret | `LAMBDA_SSH_PRIVATE_KEY` | private half of the SSH key registered in Lambda |
| variable | `LAMBDA_SSH_KEY` | that key's **name** in Lambda |
| variable | `LAMBDA_FS` | name of the persistent filesystem holding `models/` |
| variable | `VLLM_VERSION` | exact pinned vLLM version, e.g. `0.11.0` |

### 2.3 Harness — environment variables read by `run.sh` / `agent.py` / `manifest.py`

| variable | default | when to set |
|---|---|---|
| `HARNESS_ENDPOINT` | `http://localhost:8000/v1` | non-loopback endpoint ⇒ the run is billed `per_token` (API baselines) |
| `WEIGHTS_DIR` | `/persistent/models`, else `~/models` | must match what `modelctl serve` used |
| `STATE_DIR` | `<repo>/.state` | must match modelctl's — it is where `vllm-argv` (server provenance) lives |
| `HARNESS_PRICE_SNAPSHOT` | — | path to a captured `lambdactl types` output; else the live API, else `pricing/fallback-prices.json` (marks `provenance_incomplete`) |
| `HARNESS_GRADE_CONCURRENCY` | `min(cpu_count, 8)` | size of the grading pool (runs off the GPU-bound loop) |
| `HARNESS_SEED_FILE` | suite default | set automatically by `run.sh --seed-file` |
| `AGENTTASK_DATA_DIR` | `/persistent/agenttask`, `~/.harness/agenttask` | where the AgentTask data pack lives (never in git) |
| `LAMBDA_INSTANCE_ID` / `LAMBDA_REGION` / `LAMBDA_INSTANCE_TYPE` | — | exported by CI; missing ⇒ `provenance_incomplete` |
| `HARNESS_ALLOW_NETWORK=1` | off | allow the HfApi weight-revision lookup |
| `HARNESS_SKIP_WEIGHT_DIGEST=1` | off | skip hashing the weights tree — marks `nonconformant` |
| `HARNESS_SKIP_GRADING_PREFLIGHT=1` | off | start even though this host cannot grade — you will get `INFRA_GRADER` on every attempt |

### 2.4 The held constants — `harness/agent_config.json` (do not change per model)

`temperature 0.0` · `max_iters 40` · `max_tokens 8192` · `task_timeout_s 1800` ·
`seed 20260830` · `passes 3` · `max_model_len 262144` · one prompt template (`agent-v1`).
Every one of these is checked by `manifest.py build` and is a **blocking** comparability
key in `aggregate.py`. `concurrency` (default 4) is a throughput knob and is not.

---

## 3. What is measurable

Per `(model, suite)`, from `analysis/aggregate.py` (`docs/CONTRACTS.md §8`):

| metric | definition | notes |
|---|---|---|
| **cost per resolved task** — the headline | `cost_usd / resolved_attempts` where `cost_usd = active_wall_clock_h × price_per_hour` | active wall clock excludes a `--resume`'s idle gap; setup time (download + load) is reported separately, never folded in |
| resolve rate | `resolved / attempts_scored` | `attempts_scored` excludes `INFRA_*`; **includes** `SERVER_*` (an unservable model is a real cost of that model) |
| range over passes | min / max across the 3 passes | decoding is greedy, so passes measure serving nondeterminism — reported as a range, **not** a confidence interval |
| instance-level CI | cluster bootstrap over instances | legitimate: instances genuinely were sampled |
| p50 / p95 latency | per attempt wall clock | |
| tokens per task | prompt + completion, per attempt | |
| failure taxonomy | 18-value closed enum (`§4`) | `MODEL_*` (the model's fault) · `TESTS_*` (wrong answer) · `SERVER_*` (couldn't serve) · `INFRA_*` (our fault, excluded) |
| contamination view | Verified vs Pro vs AgentTask deltas per model | a large positive Verified gap is evidence the model has seen the public suite |
| setup cost | instance-hours before the first attempt | from the cost ledger, priced from the run manifest |

What makes a number **trustworthy** rather than merely present: every run carries a
manifest with the git SHA, prompt hash, weight-content digest, dependency digests,
`environment_digest` of the grader, instance id and price snapshot; every record carries the
run id and the live grader digest; bundles are checksummed and immutable. A run whose
constants differ from the others cannot reach the headline table without `--allow-mixed`.

**Three suites**, all through the identical harness:

| suite | size | role |
|---|---|---|
| SWE-bench Verified | 100 (seeded subset) | public anchor; readers can cross-check published scores; the Day-5 sanity gate |
| SWE-bench Pro | ~50 (seeded subset) | discriminator — top models spread 49–68% here vs a 79–80% cluster on Verified |
| AgentTask | 50 (internal) | uncontaminated control; publication gated on the consent decision |

---

## 4. What the deliverables are

**Per run** (in the run directory, then sealed into `<run_id>.tar.gz`):

- `run-manifest.json` — full provenance (§2); the unit of comparability
- `results.jsonl` — one `raw-result/v1` record per attempt (§3)
- `trajectories/`, `patches/`, `logs/`, `env/` — the evidence
- `SHA256SUMS`, `run-status.json` — integrity

**Committed to git** (`resultsctl` enforces the boundary): manifests, checksums, aggregate
tables, publication-safe SWE-bench samples, and the run index. **Never in git**: raw results,
trajectories, bundles, anything from the AgentTask suite beyond ids and counts.

**Publication 1 — the benchmark** (`analysis/aggregate.py` → `report/`):

- `summary.md` — paste-ready tables: headline (cost per resolved), resolve rates with
  ranges, latency, tokens, failure taxonomy, comparability table
- `summary.json`, `by_model_suite.csv`, `runs.csv`, `failures.csv`, `contamination.csv`
- the open-sourced repo itself: `modelctl`, `lambdactl`, `resultsctl`, the harness, the seed
  files and frozen partitions — enough for a reader to reproduce the exact subset and config

**Publication 2 — the fine-tune** (Phase 2, `training/`): a LoRA on Qwen3-Coder-Next trained
on Phase-1 trajectories from the `train` partition, evaluated on the untouched `final_holdout`
through the identical harness — and the tuned adapter itself.

**Standing** (Phase 3): the same harness re-run within 48 h of each major open-weight
release, from the reserve.

---

## 5. Estimated cost

Rates are Lambda on-demand from `pricing/fallback-prices.json` (what the harness bills
with when no live snapshot is captured). Run-time estimate: **~15 instance-hours per
(model, all three suites)** — 200 instances × 3 passes at concurrency 4, plus grading —
and ~1–2 h of setup (weights download + vLLM load) per dispatch. Per-hour billing rounds up.

| model | instance | $/hr | est. hours | est. run cost |
|---|---|---|---|---|
| qwen3-coder-next | 1× H100 PCIe | 2.49 | 15 | **~$40** |
| minimax-m3 | 4× H100 SXM | 15.96 | 15 | **~$240** |
| deepseek-v4-flash | 8× H100 SXM | 23.92 | 15 | **~$360** |
| glm-5.3 *(blocked on weights)* | 8× B200 | 39.92 | 15 | **~$600** |
| kimi-k3 | 8× B200 | 39.92 | 15 | **~$600** |
| **Phase 1 total, 5 models** | | | | **~$1,850** + setup ~$150 = **~$2,000** |
| qwen3.8-max *(reserve; multi-node, not CI-supported)* | 2× 8× B200 | ~80 | 18 | ~$1,450 |

Against the **$7,500 Lambda credit** (expires ~Aug 2027), the allocation is:

| item | budget |
|---|---|
| Phase 1 — core benchmark, 5 models × 3 suites | ~$2,000 |
| Phase 2 — LoRA campaign on Qwen3-Coder-Next (8–10 cycles on 8× H100) | ~$2,000 |
| Phase 3 — new-model-drop reserve (GLM-5.3 when weights land, Qwen3.8-Max, next releases) | ~$1,500 |
| reruns, mistakes, download slop | ~$900 |
| buffer, untouched | $500 |

Not on the credit: the Claude / GPT API baselines (Day 4–5 gate), **~$600–1,200 out of
pocket**, billed per token and classified `per_token` by the aggregator automatically.

**Where the money actually leaks**, and the guard for each:

| leak | guard |
|---|---|
| an instance left running | three layers: CI teardown is `if: always()` and **fails the step** if termination is unconfirmed; every instance is **leased** to its job (CI: 12 h, manual: 2 h default) and `./gpuwatch` terminates anything unleased with no harness process every 15 min; hard caps of 24 h age and $500 accrued spend on top. The reaper is safe to leave on before the secrets exist — it skips cleanly |
| discovering at the first `grade()` that the host cannot grade | grading preflight refuses before the first model call, naming the missing dependency |
| hashing hundreds of GB of weights on every dispatch | the digest cache lives with the weights on the persistent FS; hashing is parallel |
| rebuilding SWE-bench docker environments per grade | `--cache_level env` |
| the GPU idling while attempts grade | grading runs on its own pool |
| a job that cannot fit the 12 h ceiling | one suite per dispatch; the default is `swebench-verified` |
| provisioning against unfrozen partitions | `benchmark.yml` refuses before launch |

The cut order if the schedule slips: Kimi K3 → the Pro suite. If Day 10 arrives short,
publish the API baselines + Qwen alone. A narrow finished study beats a broad abandoned one.
