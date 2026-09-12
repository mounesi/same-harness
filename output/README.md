# output/ — exported runs, one folder per model, one per run

This is a working folder on disk, git-ignored except for this file (CONTRACTS.md §7.4 keeps
raw results out of git). `suites/gamespec/demo.sh` fills it at the end of every run, and
`python3 suites/gamespec/export_run.py <pulled-run-dir> [--bundle-dir …]` exports any pulled
run by hand.

```
output/
└── runs/
    ├── index.md                         every exported run, newest first
    └── <model>/                         e.g. shakedown-qwen30b
        └── <run_id>/                    e.g. shakedown-qwen30b__gamespec__20260911T090722Z__abfee0
            ├── README.md                the run on one page: model, date/time, hardware, price,
            │                            serving stack, harness, inference knobs, verdicts, floor result
            ├── metadata.json            the same as data, plus the manifest and records verbatim
            ├── game/<instance>/pass-<n>/game.html      the deliverable, rebuilt from its patch, with a
            │                                           corner badge (model, run, date); game.original.html is unmarked
            ├── game/<instance>/pass-<n>/floor-report.json
            ├── code.zip                 game/ + patches/, for handing around
            ├── run-manifest.json        full provenance (CONTRACTS.md §2)
            ├── results.jsonl            one raw-result record per attempt (§3)
            ├── patches/                 the diff the model produced, per attempt
            ├── trajectories/ logs/      the model's tool-call transcript and the harness log
            └── bundle/                  the sealed bundle, manifest and checksum
```
