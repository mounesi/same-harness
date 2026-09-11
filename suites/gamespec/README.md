# gamespec — build-a-game suite, with a human judging channel

**Status: one spec written and validated, floor check working, adapter wired in
(`harness/adapters/gamespec.py`), smoke-tested against the mock endpoint. No model run yet.**

Run it against a served model exactly like the other suites — `./harness/run.sh --model <m>
--suite gamespec --passes 1 --out ~/results` — or, from a laptop with `LAMBDA_API_KEY`
exported, as one command that brings a GPU up, runs it, pulls `game.html` back and tears
the GPU down: `./suites/gamespec/demo.sh [model]` (default `shakedown-qwen30b`, the cheapest
Qwen coder known to serve). `gamespec` is a valid single suite and deliberately **not** part
of `--suite all`, which stays the three brownfield suites the headline table is built from.
It carries its own `partitions.json` (the spec is `dev`, never Phase-2 training data) so
`suites/partitions.json` stays frozen over the three SWE-bench-shaped suites only.

A greenfield suite: give every model the same written specification and have it build a
playable game. Two things make it worth having alongside the SWE-bench-shaped suites.

**It measures the opposite skill.** SWE-bench, SWE-bench Pro and a Java migration are all
*brownfield* — read existing code, make surgical edits. This is *greenfield*: build a
coherent system from prose. Models are known to differ between the two, and a study that
only measures patching answers half the question an adopter is asking.

**It is contamination-free by construction.** We write the spec, so there is no public diff
anyone could have memorised. That is a strictly stronger position than the java17 suite,
where the `--merged-after` window manages a risk that can never be fully closed.

---

## Two channels, and they must not be blended

| | grades | verdict |
|---|---|---|
| **Automated floor** (`floor_check.py`) | does it load, is `SimCore` pure, does the physics match, does determinism hold | binary — a **floor** |
| **Human** (blind, pairwise) | is the game actually any good | ranking *above* the floor |

The floor earns its keep by protecting human attention. Without it, judges spend half a
session on builds that never launch, and "crashes on load" and "loads but is dull" collapse
into the same low score. Only submissions that clear the floor enter judging.

`Verdict.resolved` stays binary and deterministic, so **the harness itself is unchanged** —
grading is still `(task, patch, environment_digest) -> verdict` with no human in the loop.
Human judging is a *downstream consumer of the sealed bundle*: §7.4 already preserves
`patches/<instance_id>/pass-<n>.diff`, which is everything needed to rebuild a playable
artifact after the fact.

That separation is not bureaucratic. `agent.py` exits 3 when the live `environment_digest`
disagrees with the manifest; a human in that loop would break resume, reproducibility, and
the study's central claim in one move.

## Human evaluation, done so it survives review

- **Pairwise, not Likert.** People are unreliable at absolute scores and reliable at A-vs-B.
  Feed comparisons to Bradley–Terry or Elo for a ranking with real confidence intervals. A
  1–5 "fun score" will not survive a reviewer.
- **Blind, order-randomised.** Strip model identity from code comments, README text and any
  generated banner — models do sign their work. Randomise which build is on the left.
- **≥3 judges per pair, and report inter-rater agreement.** This is the step people skip.
  Without an agreement statistic, "would different judges have ranked these the same?" has
  no answer and the whole channel is dismissed.
- **Scale honestly.** The full grid is 6 models × N tasks × 3 passes. At N=20 that is 360
  builds and nobody plays 360 games. Subsample: 6 models × 3 specs × pass 0 = 18 builds,
  C(6,2)=15 pairs per spec, 45 comparisons, ×3 judges = **135 play sessions ≈ 11 hours of
  human time**. That, not GPU cost, is the budget for this suite.

**The honest tension:** the study's thesis is that the harness is the control variable, and
judges are not controllable — mood, fatigue, order effects. Above all, **human scores cannot
be regenerated**: re-running reproduces every automated number exactly and none of the human
ones. So report the two channels separately, publish the raw votes beside the aggregate, and
never compute a blended score. Framed that way it is a strength — few open benchmarks pair
deterministic grading with blind human preference over the same artifacts.

## Why a single self-contained HTML file

`specs/racing-v1.md` requires one `game.html` under 256 KB, no network, no build step. That
one decision does a lot of work: judges open a link instead of installing a toolchain,
packaging disappears, and the artifact still renders — so a winning model's game can go into
the writeup as a figure or a recording. Demand a native build instead and the project becomes
a packaging project.

## Why the floor check needs no browser

Driving real Chromium would pull Playwright into a grading environment that has already lost
a run to environment drift. Instead the spec requires a pure `SimCore` global, and
`floor_check.py` loads the page's inline scripts into plain **Node** with stubbed browser
globals. One binary, under a second.

Purity is enforced at **call** time, not load time — the renderer legitimately uses the DOM
and `Math.random`, so the file must load with permissive stubs. Once `SimCore` is captured,
the harness swaps `Math.random`, `Date.now`, `performance.now`, `document` and `window` for
traps that throw, then calls it. Global lookups resolve dynamically, so a `SimCore` that
reaches for any of them is caught while the untouched renderer is not.

## Files

| | |
|---|---|
| `specs/racing-v1.md` | the brief handed to the model — precise on simulation, silent on look and feel |
| `specs/racing-v1.reference.html` | reference implementation. Deliberately plain |
| `floor_check.py` | `python3 suites/gamespec/floor_check.py <game.html> [--json]` |

### The reference implementation is a spec-debugging tool, not a target

It was written **before** any model runs, on purpose: every place it had to guess is a place
a model would have guessed differently, and each one got pushed back into the spec. Writing
tests against an unimplemented spec is how you discover the spec was ambiguous *after* the
models have already run.

It is deliberately ugly. A polished reference would anchor judges on the very dimension the
human channel exists to differentiate.

That process already caught a bug in the floor check itself: the `MAX_SPEED` clamp check
originally ran on the standard 1000×800 track and "passed" with `speed 0` — the car had hit
a wall and been zeroed, so the clamp was never exercised. It now runs on a track large enough
that the wall is unreachable and asserts the speed actually *reaches* the cap. A check that
passes for the wrong reason is worse than no check: it reports coverage it does not have.

## What building the suite still needs

1. ~~**An adapter**~~ — done: `harness/adapters/gamespec.py` (`CONSENT_CLASS = "public"`).
   The workspace it lays down holds `SPEC.md`, a copy of `floor_check.py` and a README;
   the model's `game.html` reaches `grade()` as a new-file diff, and `grade()` always runs
   the **repo's** `floor_check.py`, never the workspace copy the model could have edited.
   `resolved` = the floor passed in full; `fail_to_pass` is the single node `floor`.
2. **More specs.** One spec is one task; the harness reports rates. Three or four specs at
   varying difficulty (racing, flight-over-city, a logistics sim to make the enterprise
   framing explicit) is the minimum for a meaningful comparison.
3. **The judging harness** — serve two builds side by side, randomise order, record votes
   keyed by `run_id` so they join back to the manifests.
4. **Budget validation.** `max_iters: 40`, `max_tokens: 8192` per response,
   `max_tool_calls_per_turn: 4` — roughly 160 tool calls. Writing a complete game in that
   budget is plausible but unproven, and it is the first thing to measure: if the median
   model runs out of iterations, the spec is too big and every score reflects the budget
   rather than the model.
