#!/usr/bin/env python3
"""floor_check.py — the automated floor for gamespec submissions.

    python3 suites/gamespec/floor_check.py path/to/game.html [--json]

Answers one question: **is this a working artifact at all?** It says nothing about whether
the game is any good — that is the human channel, and it only ever sees submissions that
clear this bar. The split is the point: without a floor, judges spend half their session
on builds that do not launch, and "crashes on load" and "loads but is dull" collapse into
the same low score.

## Why a browser is not required

Driving a real browser would drag Playwright/Chromium into the grading environment, which
is a heavy and flaky dependency for a study that has already lost a run to environment
drift. Instead the spec requires the deliverable to expose a pure `SimCore` global, and
this harness loads the page's inline scripts into plain Node with stubbed browser globals.
Node is one binary and the check takes under a second.

## How purity is actually enforced

The rendering half of a submission legitimately uses `document`, `Math.random` and timers.
`SimCore` must not. So purity cannot be checked at load time — the file has to load with
permissive stubs or nothing works.

It is checked at CALL time instead: once `SimCore` is captured, the harness replaces
`Math.random`, `Date.now`, `performance.now`, `document` and `window` with traps that
throw, then calls `create`/`step`/`hash`. Global lookups inside those methods resolve
dynamically, so a `SimCore` that reaches for any of them is caught, while the renderer —
which is not called — is untouched.

Exit codes: 0 all checks pass, 1 one or more failed, 2 usage/environment.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_BYTES = 256 * 1024

# Anything that would make the artifact depend on the network at play time. Judges may be
# offline, and a submission that phones home is not self-contained however well it runs
# today.
EXTERNAL_URL = re.compile(r"""(?:src|href)\s*=\s*["']\s*(https?:|//)""", re.I)
SCRIPT_BLOCK = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.I | re.S)
HAS_SRC = re.compile(r"\bsrc\s*=", re.I)

# The JS half. Kept here rather than in a sibling .js so the checker is one file that
# cannot drift out of sync with itself.
HARNESS_JS = r"""
'use strict';
const fs = require('fs');
const code = fs.readFileSync(process.argv[2], 'utf8');
const results = [];
const ok = (name, cond, detail) => results.push({name, ok: !!cond, detail: detail || ''});
const near = (a, b, eps) => Math.abs(a - b) <= (eps === undefined ? 1e-6 : eps);

// --- load with PERMISSIVE stubs: the renderer must be able to run without a DOM -------
const anything = new Proxy(function () {}, {
  get: () => anything, set: () => true, apply: () => anything,
  construct: () => anything, has: () => true,
});
globalThis.window = globalThis;
globalThis.document = anything;
globalThis.requestAnimationFrame = () => 0;
globalThis.addEventListener = () => {};
globalThis.HTMLCanvasElement = function () {};

let loadErr = null;
try { (0, eval)(code); } catch (e) { loadErr = e; }
ok('scripts load in a bare JS context', !loadErr, loadErr ? String(loadErr).slice(0, 200) : '');

const S = globalThis.SimCore;
ok('SimCore global is defined', !!S);
if (!S) { console.log(JSON.stringify(results)); process.exit(0); }
ok('SimCore has create/step/hash',
   typeof S.create === 'function' && typeof S.step === 'function' && typeof S.hash === 'function');
if (!(typeof S.create === 'function' && typeof S.step === 'function' && typeof S.hash === 'function')) {
  console.log(JSON.stringify(results)); process.exit(0);
}

const CONFIG = {
  seed: 12345,
  track: {width: 1000, height: 800, checkpoints: [
    {x: 100, y: 400, r: 40}, {x: 500, y: 100, r: 40},
    {x: 900, y: 400, r: 40}, {x: 500, y: 700, r: 40},
  ]},
};
const NO_INPUT = {throttle: 0, brake: 0, steer: 0};
const DT = 1 / 60;

// --- purity: install traps AFTER capture, so only SimCore's own calls are caught ------
function withTraps(fn) {
  const savedRandom = Math.random, savedNow = Date.now;
  const savedDoc = globalThis.document, savedWin = globalThis.window;
  const boom = (what) => () => { throw new Error('SimCore used ' + what); };
  Math.random = boom('Math.random');
  Date.now = boom('Date.now');
  if (globalThis.performance) globalThis.performance.now = boom('performance.now');
  globalThis.document = new Proxy({}, {get: boom('document')});
  globalThis.window = new Proxy({}, {get: boom('window')});
  try { return {value: fn()}; } catch (e) { return {error: e}; }
  finally {
    Math.random = savedRandom; Date.now = savedNow;
    globalThis.document = savedDoc; globalThis.window = savedWin;
  }
}

const pure = withTraps(() => {
  const s = S.create(JSON.parse(JSON.stringify(CONFIG)));
  return S.hash(S.step(s, NO_INPUT, DT));
});
ok('SimCore touches no browser globals or RNG', !pure.error,
   pure.error ? String(pure.error.message).slice(0, 160) : '');

// --- create() determinism -------------------------------------------------------------
const a = S.create(JSON.parse(JSON.stringify(CONFIG)));
const b = S.create(JSON.parse(JSON.stringify(CONFIG)));
ok('create() is deterministic for one config', S.hash(a) === S.hash(b));

// --- state is JSON-serialisable --------------------------------------------------------
let ser = true, serDetail = '';
try {
  const round = JSON.parse(JSON.stringify(a));
  ser = S.hash(round) === S.hash(a);
  if (!ser) serDetail = 'hash changed across a JSON round-trip';
} catch (e) { ser = false; serDetail = String(e).slice(0, 120); }
ok('state survives a JSON round-trip unchanged', ser, serDetail);

// --- step() does not mutate its argument ------------------------------------------------
const before = S.hash(a);
S.step(a, {throttle: 1, brake: 0, steer: 0.5}, DT);
ok('step() does not mutate the state it was given', S.hash(a) === before);

// --- physics, hand-computed from the spec -----------------------------------------------
// One step at full throttle from rest: v = ACCEL*dt, then *= DRAG**dt.
{
  const s0 = S.create(JSON.parse(JSON.stringify(CONFIG)));
  const s1 = S.step(s0, {throttle: 1, brake: 0, steer: 0}, DT);
  const want = (220.0 * DT) * Math.pow(0.9, DT);
  const got = s1.speed;
  ok('speed after one full-throttle step matches the spec',
     typeof got === 'number' && near(got, want, 1e-4),
     'want ' + want.toFixed(6) + ', got ' + String(got));
}
// Drag alone must bleed speed when coasting.
{
  let s = S.create(JSON.parse(JSON.stringify(CONFIG)));
  for (let i = 0; i < 60; i++) s = S.step(s, {throttle: 1, brake: 0, steer: 0}, DT);
  const fast = s.speed;
  for (let i = 0; i < 60; i++) s = S.step(s, NO_INPUT, DT);
  ok('coasting bleeds speed via drag', s.speed < fast && s.speed > 0,
     'after 1s throttle ' + fast.toFixed(2) + ', after 1s coast ' + Number(s.speed).toFixed(2));
}
// Speed is clamped and never negative under braking from rest.
{
  let s = S.create(JSON.parse(JSON.stringify(CONFIG)));
  for (let i = 0; i < 30; i++) s = S.step(s, {throttle: 0, brake: 1, steer: 0}, DT);
  ok('braking from rest never drives speed negative', s.speed >= 0, 'speed ' + String(s.speed));
}
{
  // A huge track, so the car cannot reach a wall. The first version of this check used the
  // standard 1000x800 track and "passed" with speed 0 — the car had hit a wall and been
  // zeroed, so it never exercised the clamp at all. A check that passes for the wrong
  // reason is worse than no check: it reports coverage it does not have.
  const big = {seed: 1, track: {width: 1e9, height: 1e9, checkpoints:
    [{x: 5e8, y: 5e8, r: 40}, {x: 5e8 + 1e6, y: 5e8, r: 40}, {x: 5e8, y: 5e8 + 1e6, r: 40}]}};
  let s = S.create(JSON.parse(JSON.stringify(big)));
  for (let i = 0; i < 60 * 60; i++) s = S.step(s, {throttle: 1, brake: 0, steer: 0}, DT);
  const terminal = 320.0;
  ok('speed is clamped to MAX_SPEED and actually reaches it',
     s.speed <= terminal + 1e-6 && s.speed > terminal * 0.5,
     'speed ' + Number(s.speed).toFixed(3) + ' (must be near ' + terminal + ', not 0)');
}
// Steering at rest must not rotate the car (turn rate scales with speed).
{
  const s0 = S.create(JSON.parse(JSON.stringify(CONFIG)));
  const s1 = S.step(s0, {throttle: 0, brake: 0, steer: 1}, DT);
  ok('steering at zero speed does not rotate the car',
     near(s1.heading, s0.heading, 1e-9), 'heading moved by ' + (s1.heading - s0.heading));
}

// --- checkpoint ordering: the anti-cheat ------------------------------------------------
{
  const s0 = S.create(JSON.parse(JSON.stringify(CONFIG)));
  ok('nextCheckpoint starts at 1', s0.nextCheckpoint === 1, 'got ' + String(s0.nextCheckpoint));
  ok('lap starts at 0', s0.lap === 0, 'got ' + String(s0.lap));
  // Teleport onto checkpoint 3 while 1 is still pending: must be ignored entirely.
  const cheat = JSON.parse(JSON.stringify(s0));
  cheat.x = CONFIG.track.checkpoints[3].x;
  cheat.y = CONFIG.track.checkpoints[3].y;
  const after = S.step(cheat, NO_INPUT, DT);
  ok('out-of-order checkpoint contact is ignored',
     after.nextCheckpoint === 1 && after.lap === 0,
     'nextCheckpoint ' + String(after.nextCheckpoint) + ', lap ' + String(after.lap));
  // In-order contact advances exactly one.
  const legit = JSON.parse(JSON.stringify(s0));
  legit.x = CONFIG.track.checkpoints[1].x;
  legit.y = CONFIG.track.checkpoints[1].y;
  const adv = S.step(legit, NO_INPUT, DT);
  ok('in-order checkpoint contact advances exactly one',
     adv.nextCheckpoint === 2, 'got ' + String(adv.nextCheckpoint));
}
// A full ordered circuit increments lap exactly once.
{
  let s = S.create(JSON.parse(JSON.stringify(CONFIG)));
  for (const idx of [1, 2, 3, 0]) {
    s = JSON.parse(JSON.stringify(s));
    s.x = CONFIG.track.checkpoints[idx].x;
    s.y = CONFIG.track.checkpoints[idx].y;
    s = S.step(s, NO_INPUT, DT);
  }
  ok('one full ordered circuit increments lap exactly once', s.lap === 1, 'lap ' + String(s.lap));
}

// --- determinism over a long input tape --------------------------------------------------
function tape(seed) {
  // Deterministic pseudo-inputs; nothing here may depend on Math.random.
  let x = seed >>> 0;
  const next = () => (x = (x * 1664525 + 1013904223) >>> 0) / 4294967296;
  const out = [];
  for (let i = 0; i < 600; i++) {
    out.push({throttle: next() > 0.2 ? 1 : 0, brake: next() > 0.9 ? 1 : 0, steer: next() * 2 - 1});
  }
  return out;
}
{
  const inputs = tape(99);
  const run = () => {
    let s = S.create(JSON.parse(JSON.stringify(CONFIG)));
    for (const inp of inputs) s = S.step(s, inp, DT);
    return S.hash(s);
  };
  const h1 = run(), h2 = run();
  ok('600-step tape is bit-identical across two runs', h1 === h2, h1 + ' vs ' + h2);
  ok('hash is a non-trivial string', typeof h1 === 'string' && h1.length >= 8, String(h1));
}
// A different seed must not produce the same hash — catches a hash() that ignores state.
{
  const other = JSON.parse(JSON.stringify(CONFIG));
  const s = S.create(other);
  const moved = S.step(s, {throttle: 1, brake: 0, steer: 0}, DT);
  ok('hash changes when the state changes', S.hash(s) !== S.hash(moved));
}

console.log(JSON.stringify(results));
"""


def extract_inline_js(html: str) -> tuple[str, list[str]]:
    """Concatenate every inline <script> body. Returns (code, problems)."""
    problems: list[str] = []
    parts: list[str] = []
    for attrs, body in SCRIPT_BLOCK.findall(html):
        if HAS_SRC.search(attrs):
            problems.append("a <script> tag has a src= attribute; the file must be self-contained")
            continue
        parts.append(body)
    if not parts:
        problems.append("no inline <script> content found")
    return "\n;\n".join(parts), problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game_html", help="path to the submission's game.html")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if shutil.which("node") is None:
        print("error: node is required (the floor check runs SimCore headlessly)",
              file=sys.stderr)
        return 2

    path = Path(args.game_html)
    if not path.is_file():
        print("error: %s is not a file" % path, file=sys.stderr)
        return 2

    checks: list[dict] = []

    raw = path.read_bytes()
    checks.append({"name": "game.html is under 256 KB", "ok": len(raw) <= MAX_BYTES,
                   "detail": "%d bytes" % len(raw)})

    html = raw.decode("utf-8", "replace")
    ext = EXTERNAL_URL.findall(html)
    checks.append({"name": "no external URLs (self-contained, offline-playable)",
                   "ok": not ext, "detail": "found %d external reference(s)" % len(ext)})

    code, problems = extract_inline_js(html)
    checks.append({"name": "inline <script> content is present and has no src=",
                   "ok": not problems, "detail": "; ".join(problems)})

    if code.strip():
        with tempfile.TemporaryDirectory() as td:
            js = Path(td) / "harness.js"
            src = Path(td) / "submission.js"
            js.write_text(HARNESS_JS, encoding="utf-8")
            src.write_text(code, encoding="utf-8")
            proc = subprocess.run(["node", str(js), str(src)],
                                  capture_output=True, text=True, timeout=120)
            line = (proc.stdout or "").strip().splitlines()
            if proc.returncode != 0 or not line:
                checks.append({"name": "harness ran", "ok": False,
                               "detail": (proc.stderr or "no output")[-300:]})
            else:
                try:
                    checks.extend(json.loads(line[-1]))
                except ValueError:
                    checks.append({"name": "harness produced parseable output", "ok": False,
                                   "detail": line[-1][:200]})

    failed = [c for c in checks if not c["ok"]]
    if args.json:
        json.dump({"schema": "gamespec-floor/v1", "passed": not failed,
                   "checks": checks}, sys.stdout, indent=2)
        print()
    else:
        for c in checks:
            mark = "\033[32mok\033[0m  " if c["ok"] else "\033[31mFAIL\033[0m"
            print("  %s %s%s" % (mark, c["name"],
                                 ("  — " + c["detail"]) if c.get("detail") else ""))
        print("\n%s (%d/%d)" % ("FLOOR PASSED" if not failed else "FLOOR FAILED",
                                len(checks) - len(failed), len(checks)))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.TimeoutExpired:
        print("error: the submission's scripts did not finish in 120s", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
