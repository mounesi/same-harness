# Phase 2 — LoRA fine-tune of `qwen3-coder-next`

Phase 1 benchmarks six open-weight models as agent backends with the harness held constant.
Phase 2 asks a follow-up question: *does fine-tuning the cheapest strong model on its own Phase-1
trajectories move the headline number — cost per resolved task?*

That question is only answerable if the evaluation set was never seen during training. Everything
in this directory exists to make that impossible to get wrong by accident.

```
training/
  build_dataset.py   Phase-1 trajectories -> LoRA training examples (+ dataset-manifest.json)
  train_lora.sh      8xH100 LoRA/QLoRA launcher (+ experiment-manifest.json)
  configs/           one YAML per experiment; the config IS the experiment record
```

---

## The eval rule (the only rule that matters)

> **Report the tuned model on the untouched `final_holdout` partition only.**

- `train` — the only partition `build_dataset.py` will emit examples from by default.
- `dev` — hyperparameter selection, early stopping, sanity checks. Look at it as often as you like;
  it is burned for reporting purposes the moment you tune against it.
- `final_holdout` — **write-once, look-once.** It is scored exactly once per published model, by
  `analysis/aggregate.py`, after training is finished. It never enters a dataset, never informs a
  hyperparameter, never gets "peeked at to see if it's working".

Comparability rules that go with it:

1. The tuned model is evaluated through the **same harness** as every Phase-1 model — same prompt
   template id, same iteration budget, same sampling params, same `--max-model-len`. If the harness
   changed (major `HARNESS_VERSION` bump), the base-model numbers must be re-run, not reused.
2. The base model's `final_holdout` numbers come from the same partition, so base vs tuned is a
   paired comparison on identical instances.
3. The paper reports **base and tuned side by side on `final_holdout` only**. Train/dev numbers may
   appear as training diagnostics, clearly labelled, never as headline results.
4. Fine-tuning changes the model, not the harness. A tuned run is still a run: it gets a
   `run-manifest.json`, a bundle, an index line, the works.

---

## Leakage guard — how it is enforced

Per `docs/CONTRACTS.md` §6.2, the partition split is frozen **before any run**, and the guard is
code, not a convention. `build_dataset.py`:

1. carries the frozen holdout checksum as a **compiled-in source constant**, `FINAL_HOLDOUT_SHA256`;
2. re-hashes `suites/partitions.json` on every invocation and exits `3` if it differs from either
   the constant or the checksum the file records for itself;
3. asserts every candidate example's `qualified_id` (`suite::instance_id`) is not a holdout id, and
   exits `3` **naming the offending ids** — it never filters and continues;
4. takes an **explicit list of run manifests**. A directory argument is refused outright; this tool
   does not discover results by scanning;
5. requires every input manifest's `suite.partitions_sha256` to equal the hash of the
   `partitions.json` it was handed, so two different freezes can never be mixed into one dataset;
6. refuses to write its output inside a git work tree unless the path is git-ignored — trajectory
   text (and all AgentTask content) must never enter git history.

`train_lora.sh` re-checks the same boundary from the other side before it launches anything: the
dataset's `dataset-manifest.json` must declare `split: train|dev` and carry the same
`final_holdout_sha256` that is compiled into `build_dataset.py`.

Verify the guard yourself:

```bash
python3 training/build_dataset.py --self-test     # unit tests over the guard functions
```

### One-time freeze

`FINAL_HOLDOUT_SHA256` ships as the sentinel `"UNFROZEN"`, and while it holds that value **every
build is refused** — an unpinned holdout is indistinguishable from a tampered one. When
`suites/partitions.json` is finalised (it must not be the placeholder file — the builder rejects
`"placeholder": true`), pin it exactly once and commit the change:

```bash
python3 training/build_dataset.py --freeze suites/partitions.json          # prints the line
python3 training/build_dataset.py --freeze suites/partitions.json --write  # patches this file
git diff training/build_dataset.py                                        # review, then commit
```

Re-freezing to a *different* digest is refused. Partitions are frozen once; a change needs a new
file name and a new project phase (`docs/CONTRACTS.md` §6.2).

---

## 1. Build the dataset

Run bundles are not in git. Fetch the runs you want first, then name their manifests explicitly:

```bash
./resultsctl fetch qwen3-coder-next__swebench-verified__20260830T142211Z__9f3ac1 ~/harness-data/runs
./resultsctl fetch qwen3-coder-next__swebench-pro__20260830T191455Z__2ab77c      ~/harness-data/runs

python3 training/build_dataset.py \
  --manifests ~/harness-data/runs/*/run-manifest.json \
  --partitions suites/partitions.json \
  --split train \
  --out ~/harness-data/datasets/train
```

`--manifest-list manifests.txt` (one path per line) is the reproducible alternative to a shell
glob, and is what CI should use. `--runs-root DIR` lets you pass manifests kept apart from their
run directory (e.g. `results-index/manifests/`); the run directory is then resolved by exact
`run_id`, never by scanning.

Output:

```
~/harness-data/datasets/train/
  train.jsonl              one {"messages": [...]} example per kept attempt
  dataset-manifest.json    schema lora-dataset/v1
```

`dataset-manifest.json` records `source_run_ids`, `partitions_sha256`, `final_holdout_sha256`, the
dataset's own sha256, the filters used, and the keep/skip histogram. It is the join key between an
experiment and the Phase-1 runs it was distilled from.

Useful flags: `--split dev`, `--include-unresolved` (default is resolved attempts only),
`--max-tool-chars N` (truncate long tool output; default 8000), `--max-examples N`, `--dry-run`.

**Consent:** examples inherit `consent_class` from their run. A dataset containing any `agenttask`
trajectory is `restricted` — it stays on the instance and in the restricted bucket prefix, and never
appears in git, in the paper, or in a shared artifact.

## 2. Train

```bash
./training/train_lora.sh doctor                                          # GPUs + trainer deps
./training/train_lora.sh run training/configs/qwen3-coder-next-lora-r32.yaml
```

`run` (or `manifest`, which stops after the manifest) does, in order:

1. resolve the YAML config to `config-resolved.json`;
2. run the leakage cross-check above;
3. resolve the base model through `models.d/<name>.env` and `$WEIGHTS_DIR` — the same resolution
   `modelctl` uses — and compute the **content digest of the weights directory** with the normative
   algorithm from `docs/CONTRACTS.md` §2.4 (`--no-weight-digest` skips it and marks the experiment
   `nonconformant`);
4. write `experiment-manifest.json` **before** training starts, mirroring the run manifest: repo git
   sha + dirty bit, config sha256 and fully resolved config, base-model digest and `HF_REVISION`,
   dataset sha256 + `source_run_ids` + partition checksums, every hyperparameter, GPU model/count,
   driver, and the resolved versions of torch / transformers / peft / trl / accelerate;
5. generate the trainer's own config (`trainer-config.yaml`, axolotl-shaped by default) and launch
   `torchrun --nproc_per_node=<runtime.gpus>`;
6. rewrite the manifest at the end with `status` and timings — only `status` and `timing.*` change
   between the two writes.

Everything lands in `<runtime.output_dir>/<experiment_id>/`, where `experiment_id` is
`<name>__<UTC>__<6hex>`:

```
adapter/                  LoRA weights
config.yaml               verbatim copy of the source config
config-resolved.json      what the launcher actually used
trainer-config.yaml       generated trainer input
experiment-manifest.json  schema experiment-manifest/v1
train.log
```

`--dry-run` prepares the experiment and prints the launch command without executing it.

### Config

See `configs/qwen3-coder-next-lora-r32.yaml`. It is a small YAML subset on purpose — two levels,
scalars and `[inline, lists]`, `#` comments — so the launcher stays stdlib-only and the config is
fully reproducible from the manifest. `lora.quantization: nf4` switches LoRA to QLoRA (4-bit base,
`adapter: qlora`), which is what to use if 8xH100 memory is tight at `max_seq_len: 32768`.

To drive a trainer other than axolotl, change `runtime.entrypoint` /
`runtime.entrypoint_arg_style` and the `write_trainer_config` block in `train_lora.sh` together —
they are two halves of the same decision.

## 3. Evaluate

Serve the merged/adapter model through the normal path, then run the harness exactly as Phase 1 did:

```bash
./modelctl serve qwen3-coder-next-lora        # a models.d entry pointing at the tuned weights
./harness/run.sh --model qwen3-coder-next-lora --suite swebench-verified --passes 3 --out ~/results
./resultsctl package ~/results/runs/<run_id> && ./resultsctl upload <bundle> && ./resultsctl index <bundle>
```

Then report **`final_holdout` only**, base vs tuned, with `cost_per_resolved` as the headline
(`docs/CONTRACTS.md` §8).

---

## Exit codes

| Code | `build_dataset.py` | `train_lora.sh` |
|---|---|---|
| `0` | dataset written | training finished (or manifest written) |
| `2` | usage / config error | usage / config error |
| `3` | **leakage guard tripped** | **leakage guard tripped** |
| other | — | the trainer's own exit code |

Exit `3` is never something to work around. It means the train/holdout boundary could not be
proven, and the correct response is to fix the inputs, not the guard.
