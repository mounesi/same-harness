# suites/ — what gets benchmarked, and the frozen split

Ids only. No task text, no trajectories, ever (CONTRACTS.md §6, §7.4).

| File | What it is |
|---|---|
| `verified-100.json` | `suite-seed/v1` — seeded 100-instance subset of SWE-bench Verified |
| `pro-50.json` | `suite-seed/v1` — seeded 50-instance subset of SWE-bench Pro |
| `agenttask/seed.json` | `suite-seed/v1` — all 50 internal tasks (full enumeration) |
| `partitions.json` | `partitions/v1` — the frozen train / dev / final_holdout split |
| `generate_seeds.py` | regenerates a seed file deterministically from a population id list |
| `generate_partitions.py` | regenerates the split deterministically from the seed files |
| `select.py` | the name CONTRACTS.md §6.1 uses for `generate_seeds.py`; a shim |
| `agenttask/README.md` | task format + the AI-2955 consent block — **read it first** |

## ⚠ The three committed JSON files are PLACEHOLDERS

`verified-100.json`, `pro-50.json`, `agenttask/seed.json` and therefore `partitions.json`
carry synthetic `PLACEHOLDER-*` ids, `"placeholder": true`, and a `todo` string. They exist so
the harness, the manifest writer and the leakage guard have something real-shaped to load.
**No benchmark run may use them** — `generate_seeds.py --verify` exits 1 on a placeholder and
the AgentTask adapter refuses to load one.

## Order of operations (this order is the study's integrity)

```bash
# 1. seeds — once dataset access is confirmed
python3 suites/generate_seeds.py --suite swebench-verified \
        --population verified-ids.txt --revision <dataset sha> --force
python3 suites/generate_seeds.py --suite swebench-pro \
        --population pro-ids.txt --revision <dataset sha> --force
python3 suites/generate_seeds.py --suite agenttask \
        --population agenttask-ids.txt --revision <internal task repo sha> --force

# 2. partitions — FROZEN HERE, before a single model call
python3 suites/generate_partitions.py --force
python3 suites/generate_partitions.py --print-final-holdout-sha256
#    -> compile that value into training/build_dataset.py as FINAL_HOLDOUT_SHA256

# 3. runs
./harness/run.sh --model <model> --suite <suite> --passes 3 --out ~/results
```

**Step 2 happens exactly once.** Regenerating `partitions.json` after any run invalidates the
Phase 2 leakage claim — the guarantee is that the holdout was fixed before the models ever saw
a task, and nothing downstream can verify that after the fact. If the split must change, it is
`partitions-v2.json` and a new project phase, never an edit here. `generate_partitions.py`
refuses to overwrite a non-placeholder file without `--force` for exactly this reason.

## Checks worth wiring into CI

```bash
python3 suites/select.py --verify suites/verified-100.json --population verified-ids.txt
python3 suites/select.py --verify suites/pro-50.json --population pro-ids.txt
python3 suites/generate_partitions.py --verify
```

Each re-runs the recorded algorithm and byte-compares against the committed file, so a hand
edit to any id list fails the build.
