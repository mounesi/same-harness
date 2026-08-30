# CONTRACTS.md — the spec every component is built against

**Project:** AgentTask AI-P153, "The Harness Variable" — benchmark open-weight coding models as
agent backends with the agent harness held **constant**.
**Status:** v1, frozen 2026-08-30. This document is normative. Downstream components implement
exactly what is written here; where this document and an implementation disagree, the
implementation is wrong.

Terminology: MUST / MUST NOT / SHOULD / MAY as in RFC 2119.

---

## 0. Ground rules that shape everything below

1. **The harness is the control variable.** Prompt text, iteration budget, retry policy, sampling
   params, `--max-model-len`, tool set and grading are byte-identical across models. Anything that
   varies per model is a *model* property (weights, TP/PP, instance type), never a harness property.
2. **Every run is self-describing.** `harness/run.sh` writes `run-manifest.json` *before* the first
   model call. A result without a manifest is not a result.
3. **Raw results never enter git.** Bundles go to object storage; git holds manifests, checksums,
   aggregate tables, the run index, and publication-safe samples. AgentTask consent is unresolved —
   AgentTask task text and trajectories MUST NOT enter git history under any circumstance.
4. **Leakage guard.** `suites/partitions.json` is frozen before any run.
   `training/build_dataset.py` hard-rejects every `final_holdout` id as a code invariant.
5. **Pin everything, record the resolved digest.** Version strings are not enough.

### Versioning

| Thing | Identifier | Bump rule |
|---|---|---|
| Harness | `HARNESS_VERSION` in `harness/VERSION` (semver) | patch = bugfix that cannot change a verdict; minor = new capability, verdict-neutral; **major = anything that can change a verdict** (prompt, budget, tools, sampling). Runs across a major bump are NOT comparable. |
| Result schema | `raw-result/v1` | new major version = new string, both readable by `analysis/aggregate.py` |
| Manifest schema | `run-manifest/v1` | same |
| Adapter | `ADAPTER_VERSION` per adapter module | bump on any grading change |

### Canonical JSON and hashing

- **On-disk JSON** (manifests, seed files, index): UTF-8, `indent=2`, `sort_keys=True`, one trailing
  newline. Human-diffable, because these are committed.
- **JSONL**: UTF-8, one compact object per line (`separators=(',',':')`, `sort_keys=True`), `\n`
  terminated, no blank lines. Records are append-only and MUST be flushed after each write.
- **Every `*_sha256` field** is the lowercase hex SHA-256 of the raw file bytes as written to disk,
  with no `sha256:` prefix, unless the field name says `digest` (then it carries the `sha256:` prefix
  because it may also hold a registry digest).
- **Directory digest** (used for prompts, weights) is defined once, in §2.4, and MUST be computed
  that exact way everywhere.

---

## 1. `harness/run.sh` — CLI contract

Bash, `set -euo pipefail`, usage header comment, `case` dispatch — matching `modelctl` /
`lambdactl` style. It is a thin driver: it resolves provenance, writes the manifest, then execs
`python3 harness/agent.py` once per (suite, pass).

### 1.1 Invocation

```
./harness/run.sh --model <name> --suite <name> [--passes N] [--out DIR] [options]
```

The CI contract (`.github/workflows/benchmark.yml`, do not change without a task) is exactly:

```
./harness/run.sh --model "$MODEL" --suite "$SUITE" --passes 3 --out ~/results
```

### 1.2 Flags

| Flag | Arg | Default | Meaning |
|---|---|---|---|
| `--model` | name | *required* | Must be a `models.d/<name>.env` basename. Must equal the `--served-model-name` reported by `GET <endpoint>/models`. |
| `--suite` | name | *required* | `swebench-verified` \| `swebench-pro` \| `agenttask` \| `all` |
| `--passes` | int ≥1 | `3` | Independent passes per instance. |
| `--out` | dir | `./results` | Root of the output tree (§1.5). Created if absent. |
| `--endpoint` | url | `$HARNESS_ENDPOINT` else `http://localhost:8000/v1` | OpenAI-compatible base URL. |
| `--seed-file` | path | `suites/<default for suite>` | Override the suite seed file. Recorded in the manifest. |
| `--partitions` | path | `suites/partitions.json` | Partition file used for the `partition` field on each task. |
| `--concurrency` | int | `4` | Concurrent task attempts. Affects throughput and latency percentiles, **not** verdicts. Recorded in the manifest. |
| `--task-timeout` | seconds | `1800` | Per-attempt wall-clock ceiling → `BUDGET_WALLCLOCK`. |
| `--max-iters` | int | `40` | Agent iteration budget. **Held constant.** Any non-default value sets `flags.nonconformant = true`. |
| `--resume` | run_id | — | Continue an `incomplete` run: reuses its manifest and appends only missing (instance, pass) attempts. |
| `--limit` | int | — | Debug only: first N instances in seed order. Sets `flags.truncated` and `flags.exploratory`. |
| `--instance` | id | — | Repeatable. Debug only: run just these ids. Sets `flags.exploratory`. |
| `--manifest-only` | — | — | Write the manifest(s), print the stdout lines, exit 0. No model calls, no grading. |
| `--dry-run` | — | — | `--manifest-only` plus: render and write `prompt-preview.txt` for the first task, and probe `GET <endpoint>/models`. No completions requested. |
| `-h`, `--help` | — | — | Print the usage header, exit 0. |

`--suite all` expands to three **separate runs** in this order: `swebench-verified`,
`swebench-pro`, `agenttask`. Each gets its own `run_id` and manifest; all three share one
`run_group_id`. A failure in one suite does not abort the others; the process exit code is the
numerically highest exit code across the three.

Unknown flags are a usage error. Flags are order-independent. `--` terminates flag parsing.

### 1.3 Exit codes

| Code | Name | Meaning |
|---|---|---|
| `0` | ok | Every planned attempt was executed and graded; `results.jsonl` complete; `SHA256SUMS` written; manifest `status = "complete"`. **Tasks failing to resolve is a normal `0`.** |
| `1` | usage | Bad/missing/conflicting flags. Nothing written. |
| `2` | config | Unknown model or suite, missing/invalid seed file, seed-file checksum mismatch, partitions file invalid, prompt directory missing. Nothing written. |
| `3` | preflight | Endpoint unreachable/unhealthy, served model name ≠ `--model`, weights directory missing, provenance value that is `REQUIRED` in §2 could not be resolved. Manifest written with `status = "failed"`. |
| `4` | incomplete | Started, aborted mid-flight (SIGTERM, host loss, unrecoverable server error). `results.jsonl` holds whatever completed; manifest `status = "incomplete"`. Re-runnable with `--resume`. |
| `5` | grading | More than 2% of attempts ended `INFRA_GRADER`, or the grader environment could not be constructed. Manifest `status = "incomplete"`, `flags.grading_degraded = true`. |
| `130` | interrupt | SIGINT. Same on-disk state as `4`. |

A run that exits `4`, `5`, or `130` MUST still leave a valid manifest and a checksummed partial
`results.jsonl`. Partial bundles are packaged and indexed like any other; `analysis/aggregate.py`
excludes non-`complete` runs from headline numbers unless `--include-partial` is passed.

### 1.4 stdout / stderr contract

Follows `lambdactl`: **stdout is machine-readable only, all human chatter goes to stderr.**

stdout is exactly one line per run, printed when that run terminates (any status), fields
space-separated:

```
RUN <run_id> <suite> <run_dir_abs_path> <status>
```

```
RUN qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1 swebench-verified /home/ubuntu/results/runs/qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1 complete
```

`status` ∈ `complete | incomplete | failed | manifest-only | dry-run`. Nothing else is ever written
to stdout — callers may safely `| tail -n1` or parse every line.

stderr carries `==> ` progress lines in `modelctl` style plus a per-attempt one-liner. The same
stream is teed to `<run_dir>/logs/harness.log`.

### 1.5 Directory layout written under `--out`

```
<out>/
  runs/
    <run_id>/
      run-manifest.json            # written BEFORE the first model call; rewritten once at the end
      results.jsonl                # one raw-result/v1 record per attempt (§3)
      run-status.json              # {"status","exit_code","attempts_planned","attempts_written","started_at","ended_at"}
      prompt-preview.txt           # --dry-run only
      patches/<instance_id>/pass-<n>.diff
      trajectories/<instance_id>/pass-<n>.jsonl
      logs/harness.log
      logs/attempts/<instance_id>__pass-<n>.log
      env/pip-freeze.txt
      env/nvidia-smi.txt
      env/model.env               # verbatim copy of models.d/<model>.env
      env/vllm-args.txt           # the exact argv modelctl used, from .state/vllm.log header
      SHA256SUMS                  # sha256 of every file above except itself; written last
  cost-log.jsonl                  # appended by CI, not by run.sh — reserved, do not touch
```

`instance_id` values are used as directory names verbatim; they are constrained by §6 to
`[A-Za-z0-9._-]+`, so no escaping is required.

`SHA256SUMS` is `sha256sum`-format (`<hex>  <relative path>`), sorted by path, LF endings. It is the
input to `resultsctl verify`.

### 1.6 Run id

```
<model>__<suite>__<UTC compact timestamp>__<6 hex>
qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1
```

- separator is a double underscore; model and suite names contain single hyphens and dots only
- timestamp is `date -u +%Y%m%dT%H%M%SZ` at manifest-write time
- the 6 hex chars are the first 6 of a random UUID4 — collision-safe across concurrent runs
- `run_group_id` uses the same shape with `_ALL_` in the suite position, generated once per
  `run.sh` invocation

Run ids are opaque to consumers: parse them from the manifest, never from the string.

---

## 2. `run-manifest.json` — schema `run-manifest/v1`

Written at `<run_dir>/run-manifest.json`. Written twice: once before execution with
`status: "running"` and null timings, once after with final status/timings. No other field may
change between the two writes; `analysis/` may assert this.

### 2.1 Complete example

```json
{
  "schema": "run-manifest/v1",
  "run_id": "qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1",
  "run_group_id": "qwen3-coder-next___ALL___20260830T142150Z__11b0de",
  "status": "complete",
  "created_at": "2026-08-30T14:22:11Z",
  "harness": {
    "version": "1.0.0",
    "repo_git_sha": "116df67c0a9b4f2e51d8b7a3c6e4d90f2a7b1c33",
    "repo_git_describe": "v1.0.0-0-g116df67",
    "repo_dirty": false,
    "invocation": ["./harness/run.sh", "--model", "qwen3-coder-next", "--suite", "swebench-verified", "--passes", "3", "--out", "/home/ubuntu/results"],
    "result_schema": "raw-result/v1",
    "prompt_template_id": "agent-v1",
    "prompt_dir_sha256": "3f9a1c7d84be2015c0a6f7b3d21e5c8a9d4e6f0b1a2c3d4e5f60718293a4b5c6",
    "agent_config_sha256": "a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff00",
    "adapter": "harness/adapters/swebench_verified.py",
    "adapter_version": "1.0.0",
    "adapter_sha256": "77c1d0a9f3b8e2461d5a0c9e8b7f6a5d4c3b2a190807060504030201fedcba98"
  },
  "suite": {
    "name": "swebench-verified",
    "seed_file": "suites/verified-100.json",
    "seed_file_sha256": "9e2b1a70c4d3f8567890abcdef1234567890abcdef1234567890abcdef123456",
    "selection_seed": 20260830,
    "selection_method": "seeded-uniform-without-replacement",
    "instance_count": 100,
    "instance_ids": ["astropy__astropy-12907", "django__django-11099", "…98 more…"],
    "instance_ids_sha256": "c1f0e9d8b7a6958473625140fedcba98765432100123456789abcdef01234567",
    "partitions_file": "suites/partitions.json",
    "partitions_sha256": "0a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff0"
  },
  "model": {
    "name": "qwen3-coder-next",
    "served_model_name": "qwen3-coder-next",
    "hf_repo": "Qwen/Qwen3-Coder-Next-80B-A3B-Instruct",
    "weight_revision": "8f4c1e2ab90d5f6738a1c0b9e2d3f4a5b6c7d8e9",
    "weight_revision_source": "hf_cache_metadata",
    "weight_digest": "sha256:2b7e4f8a1c093d6e5f0a7b8c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f7081920",
    "weight_file_count": 41,
    "weight_bytes": 160321548288,
    "weights_dir": "/persistent/models/qwen3-coder-next",
    "quantization": "none",
    "model_env_sha256": "5d4c3b2a19080706050403020100fedcba9876543210abcdef0123456789abcd"
  },
  "runtime": {
    "vllm_version": "0.11.2",
    "vllm_dist_digest": "sha256:aa11bb22cc33dd44ee55ff66007788990a1b2c3d4e5f60718293a4b5c6d7e8f9",
    "vllm_docker_image": null,
    "vllm_docker_image_digest": null,
    "python_version": "3.11.9",
    "torch_version": "2.7.1+cu128",
    "transformers_version": "4.57.0",
    "nvidia_driver": "570.86.10",
    "cuda_runtime": "12.8",
    "pip_freeze_sha256": "e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5",
    "requirements_lock_sha256": "1122334455667788990011223344556677889900112233445566778899001122",
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "max_model_len": 262144,
    "extra_args": "",
    "multinode": false,
    "vllm_argv": "vllm serve /persistent/models/qwen3-coder-next --served-model-name qwen3-coder-next --tensor-parallel-size 1 --max-model-len 262144 --port 8000"
  },
  "inference": {
    "endpoint": "http://localhost:8000/v1",
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
      "jitter": false,
      "retries_count_against_iteration_budget": false
    }
  },
  "hardware": {
    "instance_type": "gpu_1x_h100_pcie",
    "region": "us-west-1",
    "lambda_instance_id": "0920c5c1a1f24f2a9f4b8d7e6c5a4b39",
    "instance_name": "ci-qwen3-coder-next-19238471234",
    "hostname": "192-222-51-14",
    "gpu_model": "NVIDIA H100 PCIe",
    "gpu_count": 1,
    "gpu_memory_total_mib": 81559,
    "node_count": 1,
    "provenance": "env"
  },
  "ci": {
    "provider": "github-actions",
    "workflow": "benchmark-run",
    "github_run_id": "19238471234",
    "github_run_attempt": "1",
    "actor": "armsohi",
    "triggered_at": "2026-08-30T14:19:02Z"
  },
  "price": {
    "source": "lambdactl-types",
    "captured_at": "2026-08-30T14:19:40Z",
    "instance_type": "gpu_1x_h100_pcie",
    "price_cents_per_hour": 249,
    "currency": "USD",
    "regions_with_capacity": ["us-west-1", "us-east-3"],
    "node_count": 1,
    "effective_cents_per_hour": 249
  },
  "timing": {
    "started_at": "2026-08-30T14:22:11Z",
    "ended_at": "2026-08-30T18:41:57Z",
    "wall_clock_s": 15586,
    "attempts_planned": 300,
    "attempts_written": 300
  },
  "flags": {
    "exploratory": false,
    "truncated": false,
    "nonconformant": false,
    "grading_degraded": false,
    "resumed_from": null,
    "consent_class": "public"
  },
  "notes": ""
}
```

### 2.2 Field reference — types and how each value is obtained

`REQUIRED` means: if it cannot be resolved, `run.sh` exits `3`. `BEST-EFFORT` means: fall back to
`null` (or the documented sentinel) and set `flags.nonconformant = true`.

**`harness`**

| Field | Type | Source | Level |
|---|---|---|---|
| `version` | string | `cat harness/VERSION` | REQUIRED |
| `repo_git_sha` | string(40) | `git -C <repo> rev-parse HEAD` | REQUIRED |
| `repo_git_describe` | string | `git describe --tags --always --dirty` | BEST-EFFORT |
| `repo_dirty` | bool | `[[ -n "$(git status --porcelain)" ]]`. `true` sets `flags.nonconformant` | REQUIRED |
| `invocation` | string[] | `"$0" "$@"` as an argv array | REQUIRED |
| `result_schema` | string | literal `"raw-result/v1"` | REQUIRED |
| `prompt_template_id` | string | `harness/prompts/TEMPLATE_ID` | REQUIRED |
| `prompt_dir_sha256` | hex | directory digest (§2.4) of `harness/prompts/` | REQUIRED |
| `agent_config_sha256` | hex | sha256 of `harness/agent_config.json` (holds iteration budget, retry policy, sampling defaults) | REQUIRED |
| `adapter`, `adapter_sha256` | string, hex | path + file sha256 of the suite adapter module | REQUIRED |
| `adapter_version` | string | module constant `ADAPTER_VERSION` | REQUIRED |

**`suite`**

| Field | Type | Source | Level |
|---|---|---|---|
| `name` | enum | `--suite` | REQUIRED |
| `seed_file`, `seed_file_sha256` | string, hex | repo-relative path; sha256 of its bytes | REQUIRED |
| `selection_seed` | int | `selection.seed` in the seed file | REQUIRED |
| `selection_method` | string | `selection.method` in the seed file | REQUIRED |
| `instance_count` | int | `len(instance_ids)`; MUST equal the seed file's `count` | REQUIRED |
| `instance_ids` | string[] | copied **verbatim, in seed-file order**, from the seed file — the manifest is self-contained and readable without the repo | REQUIRED |
| `instance_ids_sha256` | hex | `sha256("\n".join(sorted(ids)) + "\n")` | REQUIRED |
| `partitions_file`, `partitions_sha256` | string, hex | `--partitions` path + sha256 | REQUIRED |

**`model`**

| Field | Type | Source | Level |
|---|---|---|---|
| `name` | string | `--model` | REQUIRED |
| `served_model_name` | string | `data[0].id` from `GET <endpoint>/models`; MUST equal `name` or exit `3` | REQUIRED |
| `hf_repo` | string | `HF_REPO` in `models.d/<model>.env` | REQUIRED |
| `weight_revision` | string \| `"unresolved"` | resolution ladder below | REQUIRED |
| `weight_revision_source` | enum `pinned_env` \| `hf_cache_metadata` \| `hf_api` \| `unresolved` | which rung of the ladder answered | REQUIRED |
| `weight_digest` | `sha256:<hex>` | content digest of the weights dir (§2.4) | REQUIRED |
| `weight_file_count`, `weight_bytes` | int | counted while digesting | REQUIRED |
| `weights_dir` | string | `$WEIGHTS_DIR/<model>` as `modelctl` computes it | REQUIRED |
| `quantization` | string | `none` \| `fp8` \| `mxfp4` \| `awq` \| … — read from the model dir's `config.json` `quantization_config.quant_method`, else `"none"` | REQUIRED |
| `model_env_sha256` | hex | sha256 of `models.d/<model>.env` | REQUIRED |

*`weight_revision` resolution ladder* (stop at the first that answers):
1. `HF_REVISION` in `models.d/<model>.env` — **the preferred method; every model env file SHOULD
   pin one.** → `pinned_env`
2. `<weights_dir>/.cache/huggingface/download/**/*.metadata` sidecars written by `hf download`:
   take the commit hash recorded there; if the sidecars disagree, treat as unresolved. →
   `hf_cache_metadata`
3. `huggingface_hub.HfApi().model_info(HF_REPO).sha` (needs network; only attempted when
   `HARNESS_ALLOW_NETWORK=1`). → `hf_api`
4. Literal `"unresolved"`, `flags.nonconformant = true`. → `unresolved`

**`runtime`**

| Field | Type | Source |
|---|---|---|
| `vllm_version` | string | `python3 -c "import importlib.metadata as m; print(m.version('vllm'))"` (equivalently `pip show vllm`) — REQUIRED |
| `vllm_dist_digest` | `sha256:<hex>` | sha256 of `<site-packages>/vllm-<ver>.dist-info/RECORD`. This pins the *installed artifact*, not just the version string — REQUIRED |
| `vllm_docker_image` | string \| null | `VLLM_DOCKER_IMAGE` from the model env (`null` when empty) |
| `vllm_docker_image_digest` | string \| null | `docker inspect --format '{{index .RepoDigests 0}}' <image>`; REQUIRED when the image is set |
| `python_version` | string | `platform.python_version()` |
| `torch_version`, `transformers_version` | string \| null | `importlib.metadata.version(...)` |
| `nvidia_driver` | string \| null | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` (first line) |
| `cuda_runtime` | string \| null | `nvidia-smi` header, or `torch.version.cuda` |
| `pip_freeze_sha256` | hex | sha256 of `env/pip-freeze.txt`, itself `python3 -m pip freeze --all` sorted |
| `requirements_lock_sha256` | hex \| null | sha256 of `harness/requirements.lock` (fully pinned, hash-pinned) |
| `tensor_parallel_size`, `pipeline_parallel_size`, `max_model_len`, `extra_args`, `multinode` | int/string/bool | sourced from `models.d/<model>.env` with `modelctl`'s defaults (`TP=1 PP=1 MAX_MODEL_LEN=262144 EXTRA_ARGS="" MULTINODE=0`) |
| `vllm_argv` | string | the `==> launching:` line from `.state/vllm.log` / `modelctl` state; `null` if unavailable |

**`inference`** — every field is read from `harness/agent_config.json` (the single source of truth
for the held-constant knobs), overridable only by the flags in §1.2. `seed` is the fixed integer
`20260830`, sent as the OpenAI `seed` parameter on every request. `temperature: 0.0` is the study
default and MUST be identical across models.

**`hardware`**

Resolution ladder, in order: (1) explicit env exported by CI; (2) `~/.harness/instance.json`
written by CI; (3) local probes; (4) `null` + `flags.nonconformant`.

| Field | Source |
|---|---|
| `instance_type` | `$LAMBDA_INSTANCE_TYPE`, else `INSTANCE_TYPE` from the model env — REQUIRED |
| `region` | `$LAMBDA_REGION` — BEST-EFFORT |
| `lambda_instance_id` | `$LAMBDA_INSTANCE_ID` — REQUIRED (this is what reconciles against Lambda billing) |
| `instance_name` | `$INSTANCE_NAME` |
| `hostname` | `hostname -f` |
| `gpu_model`, `gpu_count`, `gpu_memory_total_mib` | `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` |
| `node_count` | `2` when `MULTINODE=1`, else `1` |
| `provenance` | `env` \| `file` \| `probe` \| `unknown` |

> **CI follow-up (not this document's to fix):** `benchmark.yml` currently neither copies `harness/`
> to the instance nor exports `LAMBDA_INSTANCE_ID` / `LAMBDA_REGION` / `LAMBDA_INSTANCE_TYPE` /
> `GITHUB_RUN_ID` / the price snapshot over SSH. Both are required for this contract to hold. Track
> as a workflow task; `run.sh` MUST degrade per the ladders above rather than crash.

**`price`**

| Field | Source |
|---|---|
| `source` | `lambdactl-types` \| `snapshot-file` \| `static-fallback` |
| `captured_at` | ISO-8601 UTC when the snapshot was taken |
| `price_cents_per_hour` | int, from `regions[*].instance_type.price_cents_per_hour` in the Lambda `/instance-types` payload |
| `regions_with_capacity` | string[] at capture time (context only) |
| `effective_cents_per_hour` | `price_cents_per_hour * node_count` — this is the number `aggregate.py` uses |

Ladder: (1) `$HARNESS_PRICE_SNAPSHOT` pointing at a JSON file of the shape below (CI writes it on
the runner *before* launch, where `LAMBDA_API_KEY` lives, and scps it over); (2) run
`./lambdactl types` locally if the binary and `LAMBDA_API_KEY` are present, parsing the
`<name> $<price>/hr <regions>` line for `instance_type`; (3) `pricing/fallback-prices.json`
committed in the repo, with `source: "static-fallback"` and `flags.nonconformant = true`.

```json
{
  "schema": "price-snapshot/v1",
  "captured_at": "2026-08-30T14:19:40Z",
  "source": "lambda-api /instance-types",
  "prices": {
    "gpu_1x_h100_pcie": {"price_cents_per_hour": 249, "regions_with_capacity": ["us-west-1"]},
    "gpu_8x_h100_sxm5": {"price_cents_per_hour": 2392, "regions_with_capacity": ["us-east-3"]},
    "gpu_8x_b200_sxm6":  {"price_cents_per_hour": 3992, "regions_with_capacity": []}
  }
}
```

**`flags`**

| Field | Type | Meaning |
|---|---|---|
| `exploratory` | bool | `--limit` / `--instance` used. Excluded from all published numbers. |
| `truncated` | bool | instance list is not the full seed list |
| `nonconformant` | bool | something that could break comparability (dirty repo, non-default `--max-iters`, unresolved provenance, fallback pricing). **Set-only, never cleared.** |
| `grading_degraded` | bool | >2% `INFRA_GRADER` |
| `resumed_from` | string \| null | prior `run_id` |
| `consent_class` | `public` \| `restricted` | `restricted` for `agenttask`. Governs §7 publication rules. |

### 2.3 Manifest invariants

- Written before the first model call. If the process cannot write it, it MUST NOT call the model.
- `run_id` in the manifest == the run directory name.
- Every record in `results.jsonl` carries the same `run_id`.
- Only `status`, `timing.*`, and `flags.*` may differ between the pre-run and post-run write.

### 2.4 Directory digest (normative algorithm)

Used for `prompt_dir_sha256` and `weight_digest`. Given a root directory:

1. Walk it recursively; **skip** `.git/`, `.cache/`, `__pycache__/`, `*.pyc`, `.DS_Store`, and any
   symlink (symlinks are an error for weights → exit `3`).
2. For each remaining file compute `rel = path relative to root` (POSIX separators) and
   `h = sha256(file bytes)` as lowercase hex.
3. Sort the `(rel, h)` pairs by `rel` using byte ordering of the UTF-8 encoding.
4. Build the line stream `f"{h}  {rel}\n"` for each pair, concatenated in sorted order.
5. The digest is `sha256(that byte stream)`, hex. `weight_digest` carries the `sha256:` prefix;
   `prompt_dir_sha256` does not.

This is identical to `LC_ALL=C sort -k2` over `sha256sum` output, so it is checkable by hand.

---

## 3. Raw result records — schema `raw-result/v1`

`<run_dir>/results.jsonl`, one line per **task attempt** = one (instance_id, pass_idx) pair.
`pass_idx` is **0-based**. Exactly one record per planned attempt, including failures. Records are
written as attempts complete (order is not significant); consumers MUST sort.

```json
{
  "schema": "raw-result/v1",
  "run_id": "qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1",
  "attempt_id": "9f3ac1-astropy__astropy-12907-0",
  "suite": "swebench-verified",
  "instance_id": "astropy__astropy-12907",
  "partition": "train",
  "model": "qwen3-coder-next",
  "pass_idx": 0,
  "started_at": "2026-08-30T14:22:19Z",
  "ended_at": "2026-08-30T14:29:11Z",
  "wall_clock_ms": 412330,
  "resolved": false,
  "error_code": "TESTS_FAIL",
  "error_detail": "fail_to_pass 1/3 after patch; see logs/attempts/astropy__astropy-12907__pass-0.log",
  "tokens": {
    "prompt": 184220,
    "completion": 9841,
    "total": 194061,
    "cached_prompt": 151002
  },
  "llm_calls": 17,
  "iterations": 17,
  "tool_calls": 31,
  "harness_retries": 1,
  "latency_ms": {
    "generation_total": 268410,
    "ttft_p50": 941,
    "ttft_max": 4120,
    "per_call_p50": 12980,
    "per_call_max": 41022
  },
  "patch": {
    "present": true,
    "ref": "patches/astropy__astropy-12907/pass-0.diff",
    "sha256": "6d2a9c0b1e3f4a5b6c7d8e9f0011223344556677889900aabbccddeeff112233",
    "bytes": 4211,
    "files_changed": 2,
    "lines_added": 37,
    "lines_removed": 6
  },
  "trajectory": {
    "ref": "trajectories/astropy__astropy-12907/pass-0.jsonl",
    "sha256": "b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f70819200",
    "records": 49,
    "bytes": 918233,
    "consent_class": "public"
  },
  "grade": {
    "grader": "swebench-eval",
    "grader_version": "3.0.1",
    "adapter_version": "1.0.0",
    "fail_to_pass": {"passed": 1, "total": 3},
    "pass_to_pass": {"passed": 412, "total": 412},
    "graded_at": "2026-08-30T14:29:09Z"
  },
  "cost": {
    "gpu_seconds": 412.33,
    "effective_cents_per_hour": 249,
    "usd": 0.02852
  }
}
```

### 3.1 Field rules

| Field | Type | Rule |
|---|---|---|
| `run_id` | string | MUST match the manifest. This is the only join key. |
| `attempt_id` | string | `<run_id 6-hex suffix>-<instance_id>-<pass_idx>`; unique within a run |
| `partition` | enum | `train` \| `dev` \| `final_holdout` \| `unpartitioned`, resolved from `partitions.json` at load time |
| `pass_idx` | int | 0-based, `0 <= pass_idx < passes` |
| `wall_clock_ms` | int | `ended_at - started_at`, includes grading |
| `resolved` | bool | suite-defined success. `resolved == true` **implies** `error_code == "OK"`; the converse does not hold (`OK` + `resolved:false` is impossible — use `TESTS_FAIL`; see §4) |
| `error_code` | enum | closed enum from §4. Never free text. |
| `error_detail` | string | ≤512 chars, human-readable, no secrets, no repo source snippets |
| `tokens.*` | int | summed over every LLM call in the attempt, from the API `usage` object; retried calls **are** counted (they cost GPU time). `cached_prompt` from `usage.prompt_tokens_details.cached_tokens`, `0` if unreported |
| `llm_calls` | int | HTTP completion requests issued, retries included |
| `iterations` | int | agent loop turns consumed, retries **excluded**; `<= inference.max_iters` |
| `harness_retries` | int | retries triggered by the retry policy |
| `latency_ms.generation_total` | int | summed request duration; `wall_clock_ms - generation_total` ≈ tool/grading time |
| `patch.present` | bool | `false` ⟹ all other `patch` fields `null`, and `error_code` MUST be `NO_PATCH` or an `INFRA_*`/`BUDGET_*`/`MODEL_*` code |
| `patch.ref`, `trajectory.ref` | string | **run-directory-relative POSIX paths, never absolute** — they must survive being packaged into a bundle and unpacked anywhere |
| `trajectory.consent_class` | enum | `public` \| `restricted`; copied from the manifest. `restricted` trajectories MUST NOT be committed, quoted, or published. |
| `grade` | object \| null | `null` only when the attempt never reached grading (`INFRA_*`, `BUDGET_*` before a patch existed) |
| `cost.gpu_seconds` | float | `wall_clock_ms / 1000` |
| `cost.usd` | float | `gpu_seconds / 3600 * effective_cents_per_hour / 100`. Per-attempt cost is **attributable**, not billed; the headline number is computed from run-level wall clock in §8. |

### 3.2 Trajectory file format

`trajectories/<instance_id>/pass-<n>.jsonl`, one JSON object per agent step:

```json
{"i":3,"t":"2026-08-30T14:24:02Z","role":"assistant","kind":"tool_call","tool":"edit_file","args_sha256":"…","args_bytes":812,"content":"…","tokens":{"prompt":41022,"completion":318},"latency_ms":9210,"finish_reason":"tool_calls"}
```

Required keys: `i` (0-based step), `t`, `role` (`system|user|assistant|tool`), `kind`
(`prompt|completion|tool_call|tool_result|error`), `content` (may be truncated — record
`content_truncated_bytes` if so). Everything else is optional but MUST use these names when present.

---

## 4. Failure taxonomy — closed enum

`error_code` is one of exactly these 18 values. Adding a value is a `raw-result` major version bump.
Adapters and the agent MUST map every exception into one of them; an unmapped exception becomes
`INFRA_UNKNOWN` and is a bug.

### Success

| Code | Definition |
|---|---|
| `OK` | The attempt ran to completion, produced a patch that applied, and the grader returned all required tests passing. Always paired with `resolved: true`. |

### Model / agent failures — counted in the denominator

| Code | Definition |
|---|---|
| `NO_PATCH` | Agent terminated on its own within budget with an empty or whitespace-only diff. |
| `PATCH_MALFORMED` | A diff was produced but `git apply` (3-way, then `patch -p1`) failed. |
| `TESTS_FAIL` | Patch applied; at least one `fail_to_pass` test still fails. The ordinary "wrong answer". |
| `TESTS_REGRESSION` | Patch applied; all `fail_to_pass` pass but at least one `pass_to_pass` test broke. Reported separately because it is a distinct failure mode. |
| `BUDGET_ITERATIONS` | Hit `inference.max_iters` without terminating. Any patch present at that moment is still graded and recorded, but the code stays `BUDGET_ITERATIONS` unless it resolves (then `OK`). |
| `BUDGET_TOKENS` | Hit `inference.max_attempt_tokens`. |
| `BUDGET_WALLCLOCK` | Hit `--task-timeout`. |
| `MODEL_CONTEXT_OVERFLOW` | A request exceeded `max_model_len` even after the harness's fixed compaction step, or the server returned a context-length error. |
| `MODEL_MALFORMED_TOOL_CALL` | Tool-call payload unparseable / schema-invalid after the retry policy is exhausted. |
| `MODEL_LOOP` | The identical (tool, args_sha256) pair repeated 5 consecutive times — degenerate looping, cut short by the harness. |
| `MODEL_REFUSAL` | The model declined the task (refusal or a policy-style non-answer) instead of attempting it. |
| `MODEL_EMPTY_RESPONSE` | Empty or whitespace-only completion after the retry policy is exhausted. |

### Serving failures — counted in the denominator (they are a property of running that model)

| Code | Definition |
|---|---|
| `SERVER_ERROR` | vLLM returned 5xx, or the connection was reset mid-stream, on every retry. |
| `SERVER_UNAVAILABLE` | The endpoint stopped answering `GET /models` (OOM/crash/restart) — the attempt could not be executed. |

### Infrastructure failures — **excluded** from the denominator

| Code | Definition |
|---|---|
| `INFRA_SANDBOX` | Repo checkout, container build, or dependency setup for the task environment failed. |
| `INFRA_GRADER` | The grading harness itself crashed, hung, or returned an unparseable verdict. |
| `INFRA_HOST` | Host-level abort: SIGTERM, instance reaped, disk full, harness process killed. |
| `INFRA_UNKNOWN` | Uncategorized exception. MUST be triaged before publication. If `INFRA_UNKNOWN` exceeds 2% of attempts, the run is invalid and MUST be re-run. |

### Denominator rule (normative, used by `analysis/aggregate.py`)

```
attempts_scored = attempts where error_code NOT LIKE 'INFRA_%'
resolve_rate    = count(resolved = true) / attempts_scored
```

`SERVER_*` codes stay in the denominator: an unservable model is a real cost of that model. Runs
where `SERVER_UNAVAILABLE > 5%` MUST be flagged in the writeup.

---

## 5. Adapter interface

Every suite adapter is a module in `harness/adapters/` exposing exactly this API. No adapter may
import another adapter. Adapters MUST NOT contact the network at `load_tasks` time (seed files carry
everything needed to identify tasks; task *content* is fetched by the environment layer).

```python
# harness/adapters/<suite>.py
from __future__ import annotations
from pathlib import Path

SUITE_NAME: str          # "swebench-verified" | "swebench-pro" | "agenttask"
ADAPTER_VERSION: str     # semver; bump on ANY grading or task-shaping change
CONSENT_CLASS: str       # "public" | "restricted"   ("restricted" for agenttask)

def load_tasks(seed_file: Path) -> list[Task]: ...
def build_prompt(task: Task) -> Prompt: ...
def grade(task: Task, patch: str) -> Verdict: ...
def environment_digest() -> str: ...   # "sha256:…" identifying the grading environment
```

### 5.1 `Task`

`harness/types.py`, frozen dataclass, stdlib only.

```python
@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    suite: str                    # == SUITE_NAME
    instance_id: str              # ^[A-Za-z0-9._-]+$ , unique within the suite
    qualified_id: str             # f"{suite}::{instance_id}" — the partition key
    repo: str                     # "astropy/astropy"; "" for agenttask synthetic tasks
    base_commit: str              # git sha the agent starts from; "" if N/A
    problem_statement: str        # the issue text handed to the agent, verbatim
    fail_to_pass: tuple[str, ...] # test node ids that must go red -> green
    pass_to_pass: tuple[str, ...] # test node ids that must stay green
    environment: dict             # {"image": "...", "setup_cmds": [...], "test_cmd": "..."}
    partition: str                # "train"|"dev"|"final_holdout"|"unpartitioned"
    metadata: dict                # suite-specific, JSON-serializable, never read by the harness
    source_sha256: str            # sha256 of the canonical JSON of the upstream task record
```

Rules: `problem_statement` is passed through unmodified — no suite-specific preambles, no hints, no
formatting differences between suites. Anything an adapter wants to add goes in `metadata` and is
ignored by the prompt.

### 5.2 `Prompt`

```python
@dataclasses.dataclass(frozen=True, slots=True)
class Prompt:
    template_id: str          # MUST equal harness/prompts/TEMPLATE_ID for every suite
    system: str
    user: str
    tools: tuple[dict, ...]   # OpenAI tool schemas, identical across suites
    prompt_sha256: str        # sha256 of canonical JSON {"template_id","system","user","tools"}
    variables: dict           # what was substituted; recorded for debugging
```

**Harness-constant invariant:** `build_prompt` MUST be implemented as

```python
return harness.prompts.render(TEMPLATE_ID, {"problem_statement": task.problem_statement,
                                            "repo": task.repo, "test_cmd": ...})
```

i.e. all three adapters use the **same template id and the same template files**; only the variable
values differ. `run.sh` asserts `prompt.template_id == manifest.harness.prompt_template_id` for the
first task of every run and exits `2` on mismatch.

### 5.3 `Verdict`

```python
@dataclasses.dataclass(frozen=True, slots=True)
class Verdict:
    resolved: bool
    error_code: str                 # a value from §4
    detail: str                     # <=512 chars
    fail_to_pass: dict              # {"passed": int, "total": int}
    pass_to_pass: dict              # {"passed": int, "total": int}
    grader: str                     # "swebench-eval" | "swebench-pro-eval" | "agenttask-eval"
    grader_version: str
    raw: dict                       # grader output, JSON-serializable, goes to the attempt log
```

`grade(task, patch)` contract:
- Pure with respect to the harness: no writes outside a temp dir, no mutation of `task`.
- MUST return, never raise, for task-level failures: map them to `PATCH_MALFORMED`, `TESTS_FAIL`,
  `TESTS_REGRESSION`.
- MAY raise `GraderError` (in `harness/types.py`) only for grader-infrastructure failures; the
  caller maps that to `INFRA_GRADER`.
- `patch == ""` MUST return `resolved=False, error_code="NO_PATCH"` without building an environment.
- MUST be deterministic given (task, patch, environment_digest()).

### 5.4 Registry

`harness/adapters/__init__.py` exposes `ADAPTERS: dict[str, ModuleType]` keyed by suite name and
`get(suite: str)`. `run.sh` resolves the module path from this mapping so the manifest's
`harness.adapter` field is always the real file.

| suite | module | default seed file | consent |
|---|---|---|---|
| `swebench-verified` | `harness/adapters/swebench_verified.py` | `suites/verified-100.json` | public |
| `swebench-pro` | `harness/adapters/swebench_pro.py` | `suites/pro-50.json` | public |
| `agenttask` | `harness/adapters/agenttask.py` | `suites/agenttask/seed.json` | **restricted** |

---

## 6. Suite seed files and `partitions.json`

### 6.1 Seed file — schema `suite-seed/v1`

Committed to git. Ids only; **no task text, no trajectories.** `instance_id` MUST match
`^[A-Za-z0-9._-]+$` (it becomes a directory name).

`suites/verified-100.json`:

```json
{
  "schema": "suite-seed/v1",
  "suite": "swebench-verified",
  "frozen_at": "2026-08-30T11:04:00Z",
  "source": {
    "dataset": "princeton-nlp/SWE-bench_Verified",
    "revision": "5f1c1b3d9a0e7f2c8b4d6a1e0f9c3b7d2a5e8f10",
    "split": "test",
    "population_size": 500,
    "population_ids_sha256": "aa0f…"
  },
  "selection": {
    "method": "seeded-uniform-without-replacement",
    "seed": 20260830,
    "algorithm": "random.Random(seed).sample(sorted(population_ids), 100)  # CPython 3.11",
    "selector": "suites/select.py",
    "selector_version": "1.0.0",
    "selected_at": "2026-08-30T11:04:00Z",
    "stratified_by": null
  },
  "count": 100,
  "instance_ids": [
    "astropy__astropy-12907",
    "django__django-11099",
    "sympy__sympy-24152"
  ],
  "instance_ids_sha256": "c1f0e9d8b7a6958473625140fedcba98765432100123456789abcdef01234567"
}
```

- `instance_ids` is stored **in selection order** (reproducible from seed + algorithm).
- `instance_ids_sha256 = sha256("\n".join(sorted(instance_ids)) + "\n")` — order-independent, so it
  is a stable identity for "this set of tasks".
- `count == len(instance_ids)`; duplicates are an error.
- `selection.algorithm` is a literal, runnable one-liner. Regenerating the file MUST reproduce it
  byte for byte; `suites/select.py --verify` asserts this in CI.
- `swebench-pro` uses the same shape with `count: 50` and its own population.
- `agenttask` uses the same shape with `source.dataset: "internal/agenttask"`,
  `source.revision` = the git sha of the internal task repo, and `selection.method:
  "full-enumeration"` (all 50 internal tasks, no sampling) — `seed` is still recorded for the
  pass-level RNG.

### 6.2 `suites/partitions.json` — schema `partitions/v1`

**Frozen before any run.** Committed. Never edited after freeze; a change requires a new file name
(`partitions-v2.json`) and a new project phase. Ids are **fully qualified** (`suite::instance_id`)
so the three suites share one namespace.

```json
{
  "schema": "partitions/v1",
  "frozen_at": "2026-08-30T11:20:00Z",
  "frozen_by": "AI-P153 phase-0",
  "seed": 20260830,
  "method": "seeded-stratified-by-suite  (60/20/20 within each suite)",
  "policy": {
    "train_usable_by": ["training/build_dataset.py"],
    "dev_usable_by": ["training/build_dataset.py --split dev", "hyperparameter selection"],
    "final_holdout_usable_by": ["analysis/aggregate.py reporting only"],
    "final_holdout_write_once": true,
    "final_holdout_must_never_enter_training": true
  },
  "partitions": {
    "train":         {"count": 120, "ids": ["swebench-verified::astropy__astropy-12907", "…"]},
    "dev":           {"count": 40,  "ids": ["swebench-verified::django__django-11099", "…"]},
    "final_holdout": {"count": 40,  "ids": ["swebench-pro::pandas__pandas-51284", "…"]}
  },
  "checksums": {
    "train_sha256": "…",
    "dev_sha256": "…",
    "final_holdout_sha256": "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0",
    "all_sha256": "…"
  }
}
```

- Each `*_sha256` is `sha256("\n".join(sorted(ids)) + "\n")` for that partition.
- `all_sha256` is computed the same way over the union; the union MUST equal exactly the set of all
  ids across all three seed files — no id may be missing, none may appear twice.
- Tasks not present in any partition resolve to `partition: "unpartitioned"` at run time and are a
  validation error at freeze time.

**Leakage guard, implemented as code in `training/build_dataset.py`:**

1. A module-level constant `FINAL_HOLDOUT_SHA256 = "0f1e…e1f0"` is compiled into the source.
2. On start, recompute `final_holdout_sha256` from the loaded `partitions.json`; if it differs,
   **exit `3`** with `partitions.json final_holdout has been modified since freeze`.
3. Build the holdout id set; assert every candidate example's `qualified_id` is not in it. Any hit →
   **exit `3`**, naming the offending id. Never filter-and-continue silently.
4. `build_dataset.py` consumes an **explicit list of run manifests**
   (`--manifests m1.json m2.json …` or `--manifest-list file.txt`). It MUST NOT glob a `results/`
   directory, and MUST NOT accept a directory argument. Each manifest's
   `suite.partitions_sha256` must equal the sha256 of the `partitions.json` it was given, or exit `3`.
5. It records `source_run_ids`, `partitions_sha256`, and `final_holdout_sha256` in the dataset's own
   `dataset-manifest.json`.

---

## 7. Run bundles, `resultsctl`, and what git holds

### 7.1 Bundle

A bundle is the entire run directory, packaged immutably:

```
<run_id>.tar.gz          # tar of the run dir, top-level entry is <run_id>/
<run_id>.tar.gz.sha256   # "<hex>  <run_id>.tar.gz"
<run_id>.manifest.json   # copy of run-manifest.json, for cheap indexing without download
```

Packaging rules:
- `SHA256SUMS` inside the run dir is verified before packaging; mismatch aborts.
- Tar is created with sorted entries, `--owner=0 --group=0 --numeric-owner --mtime=@0` and gzip
  `-n`, so packaging the same directory twice yields the same bytes.
- Bundles are **write-once**. Re-packaging an existing `run_id` requires `--force` and is refused
  when the object store already holds that key.

Object-store key layout:

```
s3://$RESULTS_BUCKET/agenttask-ai-p153/runs/<suite>/<model>/<run_id>.tar.gz
                                        /runs/<suite>/<model>/<run_id>.tar.gz.sha256
                                        /manifests/<run_id>.json
```

### 7.2 `resultsctl` CLI

Bash, same style as `modelctl`/`lambdactl`. Machine-readable output on stdout, chatter on stderr.

| Command | Behaviour | stdout |
|---|---|---|
| `resultsctl package <run_dir>` | verify `SHA256SUMS`, build the deterministic tarball into `dist/` | `<bundle_path> <sha256> <bytes>` |
| `resultsctl upload <bundle>` | PUT bundle + `.sha256` + manifest copy to the object store; refuses to overwrite | `<uri> <sha256>` |
| `resultsctl index <bundle\|run_dir>` | append/update the run's line in `results-index/index.jsonl`, copy the manifest into `results-index/manifests/<run_id>.json`, write `results-index/checksums/<run_id>.sha256` | `<run_id> indexed` |
| `resultsctl verify <run_id\|bundle>` | re-download if needed; check tar sha256, then `SHA256SUMS` inside, then that every `patch.ref`/`trajectory.ref` in `results.jsonl` exists with the recorded sha256 | `<run_id> OK` / exits `1` |
| `resultsctl fetch <run_id> [dir]` | download + verify + unpack (for `analysis/`, `training/`) | `<unpacked_dir>` |

Exit codes: `0` ok, `1` verification failed, `2` usage/config, `3` object-store error.

Env: `RESULTS_BUCKET`, `RESULTS_ENDPOINT` (S3-compatible), standard AWS credential env vars.

### 7.3 `results-index/index.jsonl` — schema `run-index/v1`

**Committed to git.** One line per run; the only thing resolving `run_id` → bundle.

```json
{"schema":"run-index/v1","run_id":"qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1","run_group_id":"qwen3-coder-next___ALL___20260830T142150Z__11b0de","model":"qwen3-coder-next","suite":"swebench-verified","passes":3,"status":"complete","harness_version":"1.0.0","repo_git_sha":"116df67c0a9b4f2e51d8b7a3c6e4d90f2a7b1c33","weight_digest":"sha256:2b7e4f8a1c093d6e5f0a7b8c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f7081920","created_at":"2026-08-30T14:22:11Z","records":300,"resolved":118,"attempts_scored":297,"consent_class":"public","flags":{"exploratory":false,"nonconformant":false},"manifest_path":"results-index/manifests/qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1.json","manifest_sha256":"7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e","bundle":{"uri":"s3://harness-results/agenttask-ai-p153/runs/swebench-verified/qwen3-coder-next/qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1.tar.gz","bytes":48122934,"sha256":"3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b","packaged_at":"2026-08-30T18:44:02Z"}}
```

Append-only; a re-index of the same `run_id` replaces that line in place (`resultsctl index`
rewrites the file sorted by `created_at`). A line whose `bundle.uri` is `null` means "packaged but
not yet uploaded".

### 7.4 Git-committed vs object-store-only

| Artifact | Git | Object store | Why |
|---|---|---|---|
| `run-manifest.json` (copy under `results-index/manifests/`) | **yes** | yes | provenance must be reviewable in a PR |
| `results-index/index.jsonl` | **yes** | — | the resolver |
| `results-index/checksums/<run_id>.sha256` | **yes** | yes | tamper-evidence |
| `analysis/tables/*.csv`, `*.md` (aggregates) | **yes** | — | the paper's numbers |
| `suites/*.json`, `suites/agenttask/seed.json` (ids only) | **yes** | — | reproducibility |
| `suites/partitions.json` | **yes** | — | leakage guard |
| `pricing/fallback-prices.json` | **yes** | — | cost reproducibility |
| `samples/` — ≤5 redacted example trajectories, **SWE-bench suites only** | **yes** | yes | publication figures |
| `results.jsonl` | **no** | yes | raw results are not git content |
| `trajectories/`, `patches/`, `logs/`, `env/` | **no** | yes | size + consent |
| `<run_id>.tar.gz` | **no** | yes | the bundle |
| **anything from the `agenttask` suite beyond ids and aggregate counts** | **NEVER** | yes (restricted bucket prefix) | consent unresolved |

`.gitignore` MUST contain at minimum:

```
results/
dist/
*.tar.gz
**/trajectories/
**/results.jsonl
```

A pre-commit / CI check (`resultsctl verify --git-hygiene`) fails the build if any staged path
matches those patterns or if any staged file contains a record with
`consent_class: "restricted"`.

---

## 8. Headline metric (so everyone computes it identically)

Computed by `analysis/aggregate.py` per (model, suite), over runs with `status == "complete"` and
`flags.exploratory == false`:

```
gpu_hours          = sum(manifest.timing.wall_clock_s) / 3600          # per model+suite
cost_usd           = gpu_hours * manifest.price.effective_cents_per_hour / 100
attempts_scored    = count(records where error_code NOT LIKE 'INFRA_%')
resolved_attempts  = count(records where resolved == true)
resolve_rate       = resolved_attempts / attempts_scored
cost_per_resolved  = cost_usd / resolved_attempts                       # THE headline number
```

Reported alongside: `resolve_rate` with a bootstrap CI over the 3 passes, `pass@1` (mean over
passes) and `pass@3` (any pass resolved), the §4 failure-taxonomy histogram, median tokens
in/out per attempt, and the count of `nonconformant` runs excluded.

Instance-hour cost is charged **whole**: model download and server warm-up time are outside
`timing.wall_clock_s` and are reported separately as `setup_cost_usd` from CI's
`cost-log.jsonl` — never folded into `cost_per_resolved`, because it would penalize large models
for a one-time cache miss. State this in the paper.

---

## 9. Conformance checklist for implementers

- [ ] `bash -n` clean (`run.sh`, `resultsctl`), `python3 -m py_compile` clean (every `.py`)
- [ ] `run.sh --manifest-only` produces a manifest that validates against §2 with no `null` in a
      `REQUIRED` field, on a machine with no GPU
- [ ] every `results.jsonl` line validates against §3 and carries an `error_code` from §4
- [ ] `patch.ref` / `trajectory.ref` are relative and resolve inside the run dir
- [ ] `resultsctl verify` passes on a freshly packaged bundle
- [ ] `build_dataset.py` exits `3` on a mutated `partitions.json` and on any holdout id
- [ ] no `agenttask` trajectory, patch, or problem statement is reachable from git
