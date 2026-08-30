# analysis/ — publication tables

`aggregate.py` turns a set of run bundles into the numbers that go in the post: resolution
rate, **cost per resolved task** (the headline), latency percentiles, tokens per task, the
failure-taxonomy breakdown, and the cross-suite contamination view.

It is stdlib-only Python 3.11, reads nothing from the network, and writes nothing outside
`--out-dir`.

```bash
# the normal invocation: an explicit manifest list + unpacked bundles
python3 analysis/aggregate.py \
    --manifest-list analysis/runs-2026-09.txt \
    --results-root ~/bundles \
    --api-pricing analysis/api-pricing.json \
    --cost-log ~/results/cost-log.jsonl \
    --out-dir analysis/tables
```

---

## Inputs — explicit, never ambient

`aggregate.py` consumes an **explicit list of run manifests**. Pointing it at a directory is
refused, by design and in code:

```
$ python3 analysis/aggregate.py --manifests ~/results/runs
error: /home/…/results/runs is a directory. aggregate.py consumes an explicit list of run
manifests; globbing an ambient results/ directory is refused by design.
```

This is the same invariant `training/build_dataset.py` enforces (CONTRACTS.md §6.2 rule 4).
A published table must be traceable to a named list of runs, not to whatever happened to be
sitting on someone's disk.

| Flag | Meaning |
|---|---|
| `--manifests PATH…` | one or more `run-manifest.json` paths |
| `--manifest-list FILE…` | files with one manifest path per line; `#` comments and blank lines ignored. **Relative paths inside the list are resolved relative to the list file**, so a list file is portable. |
| `--results-root DIR…` | where unpacked bundles live, if `results.jsonl` is not next to the manifest |
| `--api-pricing FILE` | per-token rates for API-baseline models (schema below) |
| `--cost-log FILE…` | CI's `cost-log.jsonl`; setup cost is reported **separately**, never folded in |

For each manifest the tool looks for `results.jsonl` in this order: next to the manifest,
then `<root>/<run_id>/`, then `<root>/runs/<run_id>/`, then `<root>/` itself, for each
`--results-root`. That covers both "aggregate straight out of `~/results`" and "aggregate a
committed `results-index/manifests/*.json` against bundles pulled with `resultsctl fetch`".

Typical fetch-then-aggregate flow:

```bash
while read -r rid; do ./resultsctl fetch "$rid" ~/bundles; done < run-ids.txt
ls results-index/manifests/*.json > analysis/runs-2026-09.txt
python3 analysis/aggregate.py --manifest-list analysis/runs-2026-09.txt --results-root ~/bundles
```

## Validation performed before a single number is computed

- `schema` must be `run-manifest/v1` on every manifest and `raw-result/v1` on every record.
  An unrecognised schema is fatal — a future major version gets explicit support here, it is
  never silently coerced.
- `SHA256SUMS` in the run directory is verified against `results.jsonl` (and, with
  `--verify-refs`, against every `patches/` and `trajectories/` file). A mismatch is fatal.
  `--no-verify-checksums` skips it; `--require-checksums` makes a missing `SHA256SUMS` fatal.
  A `run-manifest.json` checksum mismatch is only a warning, because run.sh legitimately
  rewrites the manifest at end of run.
- Every record's `run_id` must equal the manifest's — that is the only join key (§3.1).
- Every `error_code` must be one of the 18 values in §4. Unknown codes are fatal.
- `resolved: true` must pair with `error_code: "OK"`, and `OK` must pair with `resolved: true`.
- Duplicate `(instance_id, pass_idx)` within a run is fatal.
- `timing.attempts_written` vs the record count, instance ids not in `suite.instance_ids`, and
  out-of-range `pass_idx` are warnings.

`--lenient` downgrades the per-record checks (not the checksum or schema checks) to warnings.

## Which runs are counted

Included by default only when `status == "complete"` and none of `flags.exploratory`,
`flags.nonconformant`, `flags.truncated` is set. Each escape hatch is explicit —
`--include-partial`, `--include-exploratory`, `--include-nonconformant`,
`--include-truncated` — and every excluded run is listed with its reason in `summary.md`,
`summary.json`, and `runs.csv`. §8 requires reporting the count of nonconformant runs
excluded; that is what the "Runs excluded" table is.

## The comparability guard

The study's claim is that the **harness is the control variable**. If the runs being
aggregated do not share one harness, the comparison is meaningless, so `aggregate.py`
**refuses to run** (exit 2) when either differs across runs:

- `harness.version`
- `harness.prompt_dir_sha256`

`--allow-mixed` overrides the refusal and then annotates it everywhere: a banner at the top
of `summary.md`, `comparability.mixed = true` plus the specific differences in
`summary.json`, a `harness_mixed` column on every CSV row, and a line on stderr. There is no
quiet way to mix harness versions.

Softer drift — `prompt_template_id`, `agent_config_sha256`, `adapter_version` within a suite,
`temperature`, `top_p`, `seed`, `max_iters`, `max_tokens`, `task_timeout_s`, `max_model_len` —
is reported as a warning and printed in the "Comparability drift" section. `--strict`
promotes those to fatal too.

The "Harness constants" table at the top of `summary.md` exists so a reader can see at a
glance that all of those collapsed to a single value.

## Metric definitions

Denominator rule, verbatim from §4 — `INFRA_*` attempts are excluded, `SERVER_*` attempts are
**kept** (an unservable model is a real cost of that model):

```
attempts_scored   = attempts where error_code NOT LIKE 'INFRA_%'
resolve_rate      = count(resolved = true) / attempts_scored
```

| Metric | Definition |
|---|---|
| `resolve_rate` | pooled over all attempts in the (model, suite) group |
| `pass_rate_mean` / `min` / `max` | one rate per **pass slice** — a pass slice is one `(run_id, pass_idx)` pair — then mean and range. This is the "mean + range over 3 passes" number. |
| `resolve_rate_ci95_*` | percentile bootstrap over the pass-level rates (§8). With 3 passes this interval is coarse by construction; quote the range alongside it. |
| `resolve_rate_ci95_*_instance_bootstrap` | secondary cluster bootstrap that resamples **instances** and recomputes the pooled rate. Usually the more honest interval; reported in `summary.json` only. |
| `pass_at_1` | mean of the per-pass rates |
| `pass_at_k` | share of instances resolved by **at least one** pass (`k` = number of pass slices observed) |
| `all_passes_resolved_rate` | share of instances resolved by *every* pass — a stability signal |
| `attempt_wall_s_p50` / `p95` | percentiles of per-attempt `wall_clock_ms` (includes tool time and grading) |
| `generation_s_p50` / `p95` | percentiles of `latency_ms.generation_total` |
| `ttft_ms_median_of_attempt_p50` | median across attempts of each attempt's own TTFT p50 (the raw records carry per-attempt percentiles, not raw samples) |
| tokens | median and sum of `tokens.prompt` / `completion` / `total` per attempt |

Percentiles use linear interpolation between order statistics. Bootstraps use a fixed seed
(`20260830`), so re-running the aggregator reproduces identical intervals byte for byte.

## Cost — two billing modes, one headline

**GPU-served models** (`billing_mode = instance_hours`), per §8:

```
gpu_hours         = Σ over runs of manifest.timing.wall_clock_s / 3600
cost_usd          = Σ over runs of (wall_clock_s / 3600) × price.effective_cents_per_hour / 100
cost_per_resolved = cost_usd / resolved_attempts        # THE headline number
```

The price is applied **per run**, not once per group, so runs on different instance types or
captured at different prices still sum correctly; it reduces to the §8 formula when the price
is uniform. If `price.effective_cents_per_hour` is absent it is derived from
`price_cents_per_hour × node_count` (warning). If `timing.wall_clock_s` is also missing, the
tool falls back to summing per-attempt `cost.usd` and says so loudly — that fallback
under-counts idle instance time and must not be published without a note.

**API-baseline models** (`billing_mode = per_token`) are billed from token counts instead:

```
cost_usd = Σ over attempts of
             (tokens.prompt − tokens.cached_prompt)/1e6 × input_usd_per_mtok
           + tokens.cached_prompt/1e6            × cached_input_usd_per_mtok
           + tokens.completion/1e6               × output_usd_per_mtok
```

A model is treated as API-baseline when `price.billing_mode == "per_token"` in its manifest,
or the model name appears in the `--api-pricing` file, or the manifest carries neither a
price-per-hour nor a `lambda_instance_id`. If a model resolves to per-token billing and no
rates are available, the tool **fails** (exit 2) rather than silently reporting a free model.

`--api-pricing` file (schema `api-pricing/v1`; `cached_input_usd_per_mtok` defaults to the
input rate when omitted):

```json
{
  "schema": "api-pricing/v1",
  "captured_at": "2026-08-30T12:00:00Z",
  "currency": "USD",
  "models": {
    "some-api-baseline": {
      "provider": "vendor",
      "input_usd_per_mtok": 3.0,
      "cached_input_usd_per_mtok": 0.30,
      "output_usd_per_mtok": 15.0
    }
  }
}
```

Rates may also be inlined in a manifest's `price` object (`billing_mode: "per_token"` plus the
same three keys); the manifest wins over the file.

**Setup cost is never folded into the headline.** Model download and server warm-up happen
outside `timing.wall_clock_s`; `--cost-log` reads CI's `cost-log.jsonl` and reports
`setup_cost_usd` in its own column, per §8, so a large model is not penalised for a one-time
cache miss. Records are matched by `run_id` and may carry either `setup_cost_usd` /
`setup_usd`, or `setup_seconds` / `setup_s` / `download_s` plus a cents-per-hour field.

`attributable_cost_usd` (the sum of per-attempt `cost.usd`) is carried in `summary.json` as a
cross-check only — per §3.1 that number is *attributable*, not billed.

## Cross-suite contamination view

Same model, same harness, three suites. SWE-bench Verified predates these checkpoints and is
plausibly in their pretraining data; SWE-bench Pro and the internal AgentTask suite are not.
So a **large positive Verified gap is the contamination signal**:

```
verified_gap_vs_mean_others = verified_resolve_rate − mean(pro, agenttask)
```

`contamination_flag` is set when that gap is at least `--contamination-threshold` (default
`0.10`, i.e. 10 percentage points). The table also carries the pairwise deltas and a
`complete_triple` column — a model missing a suite is shown for completeness but must not be
quoted as contamination evidence.

The threshold is a reporting convenience, not a test. It is a descriptive flag over three
suites with different task counts and difficulty; the deltas, not the flag, are what belongs
in the prose.

## Outputs

Written to `--out-dir` (default `analysis/tables`):

| File | Contents |
|---|---|
| `summary.md` | paste-ready markdown: harness constants, headline, resolution detail, latency/tokens, failure taxonomy, contamination, per-model rollup, run inventory, exclusions, warnings |
| `summary.json` | schema `aggregate-report/v1` — every number above plus provenance, options, and diagnostics |
| `by_model_suite.csv` | one row per (model, suite) |
| `failures.csv` | long form: one row per (model, suite, error_code) over all 18 codes |
| `contamination.csv` | one row per model |
| `runs.csv` | run inventory including excluded runs and their reasons |

`--print md|json|none` controls what also goes to stdout (default `md`). All progress and
warnings go to stderr, matching `modelctl` / `lambdactl`. `--no-write` suppresses files.

### Git hygiene

These outputs are aggregates and are **meant to be committed** (§7.4:
`analysis/tables/*.csv`, `*.md` → git). They contain no task text, no problem statements, no
patches and no trajectories. Diagnostics may quote a suite instance id, which is git-safe —
ids are already committed in `suites/` — but nothing else from a restricted run reaches these
files. `summary.json` records each run's `consent_class` so a reviewer can see that AgentTask
runs contributed counts only.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | tables produced (warnings may still have been printed) |
| `1` | usage error |
| `2` | validation or comparability refusal — mixed harness without `--allow-mixed`, bad schema, checksum mismatch, unknown error code, unresolvable cost, no eligible runs |
| `3` | missing or unreadable input (manifest not found, no `results.jsonl`, directory argument, unwritable `--out-dir`) |
| `130` | interrupted |

## Options quick reference

```
--allow-mixed              aggregate across differing harness versions / prompt hashes,
                           loudly annotated everywhere
--strict                   treat soft comparability drift as fatal too
--include-partial          include runs whose status is not "complete"
--include-exploratory      include --limit/--instance debug runs
--include-nonconformant    include runs flagged nonconformant
--include-truncated        include runs with a truncated instance list
--no-verify-checksums      skip SHA256SUMS verification
--require-checksums        fail when a run has no SHA256SUMS
--verify-refs              also verify every patches/ and trajectories/ file
--lenient                  per-record validation failures become warnings
--bootstrap-iters N        resamples for the CI (default 10000; 0 disables)
--contamination-threshold  Verified-gap fraction that raises the flag (default 0.10)
```

## Assumptions a reviewer should check

1. **`api-pricing/v1` is defined here, not in CONTRACTS.md.** The spec does not cover
   API-baseline models; the schema above and the billing-mode detection ladder are this
   component's proposal. If the contract later names a different shape, this is the file to
   change.
2. **`cost-log.jsonl` field names are inferred.** CONTRACTS.md §1.5 reserves the file for CI
   and does not specify its schema. The reader accepts several plausible key names and skips
   what it cannot parse; confirm against whatever CI actually writes.
3. **Several runs of the same (model, suite) are pooled**, with each run's own price applied
   to its own wall clock. Pass slices stay distinct — a pass is `(run_id, pass_idx)`, so
   pooling two 3-pass runs yields six pass slices, not three.
4. **`pass@k` uses `k` = the number of pass slices observed**, which is 3 for a standard run
   and larger when runs are pooled. The column header says so; the prose should too.
5. **The §8 bootstrap is over pass-level rates**, as written in the contract. With three
   passes that CI is essentially the min/max. The instance-cluster CI in `summary.json` is the
   one worth quoting if the writeup wants a real interval — that choice is not settled by the
   contract.
