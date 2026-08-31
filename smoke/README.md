# smoke — prove the pipeline works before paying for it

```bash
./smoke/run-smoke.sh          # ~10s, no GPU
./smoke/run-smoke.sh --keep   # keep the run directory for inspection
```

Three pieces:

| file | what it is |
|---|---|
| `mock_endpoint.py` | a fake OpenAI-compatible server. Speaks `/v1/models` and `/v1/chat/completions`, and drives a scripted read → edit → finish tool loop. `MOCK_MODE=solve\|noop\|flaky`. |
| `make_pack.py` | builds a synthetic 2-task AgentTask pack: a real python package with a real bug (`divide()` raises on a zero denominator) and real hidden tests. Written outside the repo — a data pack is task content, and CONTRACTS §7.4 keeps that out of git. |
| `run-smoke.sh` | runs the whole path and asserts nine things about it. |

## Why it exists

Every expensive bug the reviews found was a *plumbing* bug — `harness/` never copied to the
instance, `WEIGHTS_DIR` exported to one step but not the next, a cost log keyed on an id the
aggregator never reads. None of them are visible to a syntax check. All of them would have
surfaced on the first real run, after the weights download and the vLLM load, with an 8×B200
meter running at ~$54/hr.

A GPU is not required to catch that class of bug. A server that speaks the same protocol is.

## What it caught on its first run

1. **Every agenttask attempt would have scored `NO_PATCH`.** `capture_patch()` derives the
   diff from `git diff --cached`, but the adapter materializes a plain tarball with no
   repository — so the diff was always empty, regardless of what the model did. The whole
   uncontaminated control suite would have reported 0% for every model and read as a
   *finding about the models* rather than broken plumbing. Fixed by giving an
   adapter-materialized workspace a git base (`harness/agent.py:_ensure_git_base`).

2. **The grading preflight passed on a host that could not grade.** It checked docker and
   the swebench module, but agenttask grades by running pytest — so with pytest absent it
   reported `PREFLIGHT_OK=1` and then graded every attempt `TESTS_FAIL 0/N`. Adapters now
   declare their own host requirements (`grading_requirements()`), and the preflight
   consults them.

Both are the same shape as the bugs review found: an infrastructure failure wearing the
costume of a model result.

## What it asserts

1. mock endpoint answers `/v1/models` — run.sh preflight tier 1
2. the manifest builds and every REQUIRED provenance field resolves — tier 2
3. grading dependencies are present — tier 3
4. attempts execute; one record per attempt in `results.jsonl`
5. every record carries the run id, and the `environment_digest` the manifest recorded (§2.3)
6. the solved task grades `resolved=true` — proving the grader ran and can tell a real fix
   from a wrong one (verified separately: a wrong patch grades `TESTS_FAIL`)
7. `aggregate.py` produces the headline table — and *refuses* to until
   `--include-nonconformant` is passed, because a smoke run deliberately deviates. Being
   forced to pass that flag is the flag split proving it will not silently publish a
   non-study run.
8. `resultsctl` packages a bundle whose checksums verify
9. the leakage guard fails closed when the holdout boundary is not frozen, and its 11-test
   rejection suite passes

## Limits

This proves the pipeline is *wired*. It does not prove a model is any good, that vLLM serves
correctly at 8-way tensor parallelism, or that the SWE-bench adapters grade real instances —
those need the real thing. It is the check that makes the first real run likely to succeed,
not a substitute for it.
