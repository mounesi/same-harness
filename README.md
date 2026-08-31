# nvidia-lambda — model serving for "The Harness Variable" (AI-P144)

One CLI, one OpenAI-compatible endpoint, six open models. The harness always talks to
`http://localhost:8000/v1`; which model answers is just `./modelctl switch <name>`.

## Quick start (on a Lambda instance)

```bash
pip install vllm hf_transfer
./modelctl list                     # see registry + what's on disk
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
| qwen3-coder-next | 1× H100 | cheap anchor |
| minimax-m3 | 4–8× H100 | verify specs on HF card before reserving |
| deepseek-v4-flash | 8× H100 | use the retrained agentic checkpoint |
| glm-5.3 | 8× B200 | BLOCKED on Zhipu weights release — no 5.2 fallback; if not out by run day, defer to reserve |
| kimi-k3 | 8× B200 (single node) | native MXFP4; accept HF license first |
| qwen3.8-max | 2× 8× B200 (multi-node) | native FP8; needs Ray cluster; accept HF license first |

## Conventions (held constant across models — this is the study)

- `--max-model-len 262144` for every model
- weights live in `WEIGHTS_DIR` (defaults to `/persistent/models` if `/persistent`
  exists — use a Lambda persistent filesystem so weights survive instance restarts)
- pin the vLLM version in your run notes; Kimi K3 may need Moonshot's official
  docker image (uncomment `VLLM_DOCKER_IMAGE` in its env file)

## TODO before Day 1

- [ ] verify every `HF_REPO` id (marked TODO in models.d/)
- [ ] accept Kimi K3 + Qwen3.8-Max licenses on HF
- [ ] resolve MiniMax M3 spec conflict (~230B vs 428B) → set TP=4 or 8
- [ ] dry-run the Ray two-node launch for qwen3.8-max

## GPU lifecycle — lambdactl + CI

`lambdactl` drives the Lambda Cloud API (needs `LAMBDA_API_KEY`):

```bash
./lambdactl types                    # instance types + live availability
./lambdactl up gpu_8x_b200_sxm6      # launch, wait for active + SSH, print "id ip"
./lambdactl ls
./lambdactl down --all               # kill everything
./lambdactl reap 24                  # kill anything older than 24h
```

CI (`.github/workflows/`):

- **benchmark.yml** — `workflow_dispatch(model, instance_type, suite)`: launch instance →
  attach persistent weights filesystem → `modelctl serve` → run harness (3 passes) →
  upload results + instance-hours cost log → **teardown in an `if: always()` step**.
  A `concurrency: gpu-run` group means only one GPU run at a time.
- **reaper.yml** — every 6h, terminates any instance alive >24h. The credit's seatbelt.

Secrets: `LAMBDA_API_KEY`, `LAMBDA_SSH_PRIVATE_KEY`. Variables: `LAMBDA_SSH_KEY`, `LAMBDA_FS`.
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
