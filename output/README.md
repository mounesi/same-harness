# output/ — exported runs, one folder per model, one per run

This is a working folder on disk, git-ignored except for this file (CONTRACTS.md §7.4 keeps
raw results out of git). `suites/gamespec/demo.sh` fills it at the end of every run, and
`python3 suites/gamespec/export_run.py <pulled-run-dir> [--bundle-dir …]` exports any pulled
run by hand.

```
output/
└── runs/
    ├── index.md / index.html            every exported run, grouped by model, newest first
    ├── _compare/<instance>__<run>_vs_<run>/index.html    from compare_runs.py — side by side
    └── <model>/                         e.g. shakedown-qwen30b
        └── <run_id>/                    e.g. shakedown-qwen30b__gamespec__20260911T090722Z__abfee0
            ├── index.html               the click-through page: config, attempts, floor result, Play
            ├── README.md                the same on one page, in Markdown: model, date/time, hardware,
            │                            price, serving stack, harness, inference knobs, verdicts, floor
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

## Browsing and comparing, without touching a GPU

```bash
./suites/gamespec/serve.sh              # http://localhost:8787/index.html — click through everything
```

`output/runs/index.html` lists every export grouped by model; each run's own page has a
**Play** link straight into its badged `game.html`, its full configuration, and its floor
result. Opening it costs nothing — this only reads what's already on disk.

To put two (or more) runs of the *same spec* side by side — different models, or two
attempts of the same model — export both, then:

```bash
python3 suites/gamespec/compare_runs.py <run_id_a> <run_id_b> [--instance racing-v2]
```

Bare run ids resolve under `output/runs/`; `--instance` is only needed when the runs share
more than one spec. Writes `output/runs/_compare/.../index.html`: one column per run, each
playable, with its configuration, attempt stats and failed floor checks lined up beside
the others so a difference in outcome can be read against a difference in setup.
