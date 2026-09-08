# java17 — Java 8 → 17 migration suite (DESIGN, not yet built)

An enterprise-facing suite for the Harness Variable study: *can an open coding model
perform a JDK 8 → 17 migration?* Chosen because it is the migration with the largest
budget line attached in real organisations, and because it is gradeable without writing
correctness tests by hand — a merged migration PR carries its own oracle, since the
project's test suite had to go green under JDK 17 for the PR to land.

**Status: feasibility investigated, design open on one fork. No tasks exist yet.**
Nothing here has run against a model. The blocking question is not the adapter, it is
where 40 tasks come from — see *Where the tasks come from*, which is the part that is
genuinely unresolved.

---

## Why this suite (and why it fits the harness)

- **The oracle is inherited, not authored.** `pass_to_pass` is the project's own test
  suite; `fail_to_pass` is whatever only passes once the migration is done. We do not
  write assertions about "good migration", we run the tests the maintainers already wrote.
- **The hidden-test overlay does the right thing for free.** §5 grading applies hidden
  tests *after* the agent's patch, so a model cannot delete or weaken tests to pass. For
  a migration this is exactly right: overlay the *post*-migration test sources onto the
  *pre*-migration tree and the main code is forced to match.
- **It exercises what actually varies between models** — reading unfamiliar code, making
  many small consistent edits, and not breaking things it was not asked to touch.

Real content of a JDK 8 → 17 migration, for reference: `javax.*` → `jakarta.*`, removed
JAXB/JAX-WS/CORBA/Nashorn, strong encapsulation of JDK internals (JEP 396/403) and the
`--add-opens` escape hatch, `Class.newInstance()` → `getDeclaredConstructor().newInstance()`,
Maven `source`/`target` → `release`, and forced version bumps of every bytecode-touching
library (Lombok, Mockito, Byte Buddy, ASM).

---

## Where the tasks come from — the unresolved part

`mine_candidates.py` harvests merged migration PRs. Running it produced two findings that
change the design, both cheap to get and worth recording before anyone builds on them.

**1. The obvious query is wrong by two orders of magnitude.** Asking for `"java 17"` *and*
`migrate` in the title returns **63** merged PRs since 2026-01-01, which reads as "there is
not enough material here". Real migrations are named after their visible consequence
instead:

| query (merged ≥ 2026-01-01, `language:Java`) | PRs |
|---|---|
| `"java 17"` + `migrate` in title | 63 |
| `"jdk 17"` OR `"java 17"` in title | 2,308 |
| `jakarta` in title | 2,815 |
| `"spring boot 3"` in title | 6,172 |

The pool is ample. The script now runs the broad query set.

**2. Pool size is not the constraint — diff shape is.** Filtering to quality repos in the
contamination-safe window returns either:

- *mature projects doing small follow-ups* — spring-boot (81k stars) **1 file**,
  apache/maven **2 files**, apache/parquet-java **2 files**. These are not migrations;
  those projects migrated years ago.
- *or whole-monorepo migrations far too large for the budget* — `iti-ict/wakamiti` PR#387
  changed **759 files** (+30,726/−18,273). No agent completes that in `max_iters: 40`.

The Goldilocks case — a 10–50 file whole-project migration, in a repo worth citing,
merged recently enough to be contamination-safe — is **rare**, because by 2026 the
projects worth citing have already migrated.

### The fork (needs a decision before any adapter is written)

**A. Decompose large migrations into module-scoped tasks.** One 759-file monorepo
migration is not one task, it is potentially twenty: `module-a/` migrated independently,
graded by that module's own tests. This inverts the problem — instead of 40 repos you need
a handful of large migrations. Best realism, most mining machinery, and the module
boundary has to genuinely isolate (a module whose compilation depends on an unmigrated
sibling is not independently gradeable).

**B. Synthesise by back-porting.** Take a project already on 17, mechanically rewrite one
module to Java 8 idioms (jakarta→javax, `var` expanded, records→classes, build config
downgraded), and ask the model to re-migrate. Contamination-free *by construction*,
unlimited supply, difficulty dialable. Costs realism: a mechanical back-port is not what
legacy code actually looks like, and a reviewer will say so.

**C. Widen the date window and mitigate.** Far more Goldilocks candidates exist in
2021–2024, but every evaluated model may have trained on those diffs. Mitigate by
perturbing snapshots (rename symbols, restructure packages). Weakest option: perturbation
is unfalsifiable hand-waving unless you can show it defeats memorisation.

**A is the recommendation**, with **B** as a supplement if A yields too few isolated
modules. The decision belongs to whoever will defend the methodology, not to the miner.

---

## Contamination

The reason `--merged-after` exists and defaults to a recent date. If a migration landed
before a model's training cutoff, that model may have memorised the diff and the task
measures recall, not capability — the single most common way a coding benchmark is
dismissed. The chosen date is written into the miner's output and belongs in the run's
provenance alongside the seed and the partition hash.

Note this is a **different** leak from the one `training/build_dataset.py` guards. That
guard keeps *our* holdout out of *our* fine-tuning set. This one concerns what the vendor
put in the model before we ever saw it, and nothing in this repo can detect it — only the
selection rule prevents it.

---

## What building it requires (once the fork is decided)

1. **A `java17` adapter** implementing CONTRACTS.md §5 — `load_tasks`, `build_prompt`,
   `materialize`, `grade`, `environment_digest`, `grading_requirements` — registered in
   `harness/adapters/__init__.py` (`SUITES`, `MODULE_NAMES`). Copy `agenttask.py`'s
   structure but set `CONSENT_CLASS = "public"`: this suite is built from public repos, and
   inheriting `"restricted"` would make `resultsctl --git-hygiene` refuse to stage results
   that are perfectly publishable.
2. **A JUnit-XML grader.** Maven Surefire writes `target/surefire-reports/TEST-*.xml` and
   Gradle writes the equivalent — structured, per-test-method, and far more reliable than
   the stdout scraping `_parse_pytest_output` has to do. This is a real advantage over the
   Python suites, not a chore.
3. **A prepared image**: JDK 17 + Maven/Gradle + a **pre-warmed `~/.m2`**. Not optional —
   `agent_config.json` caps a single tool call at `command_timeout_max_s: 600`, and a cold
   Maven dependency resolution on a large project can exceed that on its own. The agent
   would fail on infrastructure, and it would look like a model failure.
4. **Budget validation.** `max_iters: 40`, `task_timeout_s: 1800`, `max_tokens: 8192` per
   response. A module migration that needs 200 iterations does not fit, and raising the
   limit marks the run `nonconformant`. Task scoping must respect the budget, or the
   budget must be forked deliberately as a separate harness version — never mixed.

## Cost, roughly

40 tasks × 3 passes = 120 attempts/model. At `concurrency: 4` and ~8 min average, ≈4 GPU-h
per model; six models ≈ 24 GPU-h. That is ~$80 if it were all 1×H100 — but the large models
are $53.52/h, so realistically **$1,000–1,500** for one sweep of this suite. Against a
$7.5k credit with three suites already planned, adding a fourth is a budget decision.
