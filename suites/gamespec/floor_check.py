#!/usr/bin/env python3
"""floor_check.py — the automated floor for gamespec submissions.

    python3 suites/gamespec/floor_check.py path/to/game.html [--json] [--spec racing-v2]

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
HARNESS_PRELUDE_JS = r"""
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
"""

# racing-v1 — see specs/racing-v1.md §5.
HARNESS_V1_JS = r"""
const S = globalThis.SimCore;
ok('SimCore global is defined', !!S);
if (!S) { console.log(JSON.stringify(results)); process.exit(0); }
ok('SimCore has create/step/hash',
   typeof S.create === 'function' && typeof S.step === 'function' && typeof S.hash === 'function');
if (!(typeof S.create === 'function' && typeof S.step === 'function' && typeof S.hash === 'function')) {
  console.log(JSON.stringify(results)); process.exit(0);
}

// Everything below CALLS the submission. A SimCore that throws — the first live run's
// step() read a `config` that was not in scope — must come out as a failed check with the
// message, not as a node stack trace and a bare "harness ran: false". Only calls inside
// withTraps() were guarded before; the determinism and physics sections were not.
try {

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
} catch (e) {
  ok('SimCore ran every floor check without throwing', false,
     String(e && e.stack ? e.stack.split('\n').slice(0, 2).join(' | ') : e).slice(0, 300));
}

console.log(JSON.stringify(results));
"""

# racing-v2 — see specs/racing-v2.md §6. Every constant below is a spec constant.
HARNESS_V2_JS = r"""
const S = globalThis.SimCore, V = globalThis.View;
ok('SimCore global is defined', !!S);
ok('View global is defined', !!V);
if (!S || !V) { console.log(JSON.stringify(results)); process.exit(0); }
const fns = (o, names) => names.every(n => typeof o[n] === 'function');
ok('SimCore has create/step/hash', fns(S, ['create', 'step', 'hash']));
ok('View has project', fns(V, ['project']));
if (!fns(S, ['create', 'step', 'hash']) || !fns(V, ['project'])) {
  console.log(JSON.stringify(results)); process.exit(0);
}

const TRACK = {width: 1000, height: 800, roadWidth: 120,
  centerline: [{x: 150, y: 150}, {x: 850, y: 150}, {x: 850, y: 650}, {x: 150, y: 650}],
  checkpoints: [{x: 150, y: 400, r: 40}, {x: 500, y: 150, r: 40}, {x: 850, y: 400, r: 40}, {x: 500, y: 650, r: 40}]};
const CONFIG = {seed: 12345, laps: 3, opponents: 2, track: TRACK};
const cfg = (over) => JSON.parse(JSON.stringify(Object.assign({}, CONFIG, over || {})));
const withTrack = (over) => cfg({track: Object.assign({}, TRACK, over)});
const NO_INPUT = {throttle: 0, brake: 0, steer: 0, boost: 0};
const FULL = {throttle: 1, brake: 0, steer: 0, boost: 0};
const BOOST = {throttle: 1, brake: 0, steer: 0, boost: 1};
const DT = 1 / 60;
const cp = (s) => JSON.parse(JSON.stringify(s));

function withTraps(fn) {
  const savedRandom = Math.random, savedNow = Date.now;
  const savedDoc = globalThis.document, savedWin = globalThis.window;
  const boom = (what) => () => { throw new Error('SimCore/View used ' + what); };
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

try {
const pure = withTraps(() => {
  const s = S.create(cfg());
  V.project({x: 1, y: 2, z: 0}, {x: 0, y: 0, z: 10, yaw: 0.3, pitch: -0.3, fov: 1, width: 800, height: 600});
  return S.hash(S.step(s, BOOST, DT));
});
ok('SimCore and View touch no browser globals or RNG', !pure.error,
   pure.error ? String(pure.error.message).slice(0, 160) : '');

// --- create(): determinism, shape, serialisability, no mutation ------------------------
const a = S.create(cfg()), b = S.create(cfg());
ok('create() is deterministic for one config', S.hash(a) === S.hash(b));
{
  const s = a, p = s.player;
  ok('initial state carries the required fields',
     s.t === 0 && s.laps === 3 && s.finished === false && s.lap === 0 && s.nextCheckpoint === 1 &&
     s.lapStart === 0 && Array.isArray(s.lapTimes) && s.lapTimes.length === 0 && s.bestLap === null &&
     !!p && p.speed === 0 && p.boostCharge === 1 && p.boostActive === 0 &&
     Array.isArray(s.opponents) && s.opponents.length === 2 && Array.isArray(s.ranking),
     JSON.stringify({t: s.t, laps: s.laps, finished: s.finished, lap: s.lap, nextCheckpoint: s.nextCheckpoint,
                     lapStart: s.lapStart, lapTimes: s.lapTimes, bestLap: s.bestLap, player: p,
                     opponents: (s.opponents || []).length, ranking: s.ranking}).slice(0, 200));
  ok('every car starts at checkpoint 0 with speed 0, heading toward checkpoint 1',
     near(p.x, 150) && near(p.y, 400) && near(p.heading, Math.atan2(150 - 400, 500 - 150), 1e-9) &&
     s.opponents.every(o => near(o.x, 150) && near(o.y, 400) && o.speed === 0 && o.lap === 0 && o.nextCheckpoint === 1));
}
{
  let ser = true, detail = '';
  try { ser = S.hash(cp(a)) === S.hash(a); if (!ser) detail = 'hash changed across a JSON round-trip'; }
  catch (e) { ser = false; detail = String(e).slice(0, 120); }
  ok('state survives a JSON round-trip unchanged', ser, detail);
}
{
  const before = S.hash(a);
  S.step(a, BOOST, DT);
  ok('step() does not mutate the state it was given', S.hash(a) === before);
}

// --- physics, hand-computed from the spec -----------------------------------------------
{
  const s1 = S.step(S.create(cfg()), FULL, DT);
  const want = (220.0 * DT) * Math.pow(0.9, DT);
  ok('on the road: speed after one full-throttle step matches the spec',
     near(s1.player.speed, want, 1e-4), 'want ' + want.toFixed(6) + ', got ' + String(s1.player.speed));
  ok('t advances by dt each step', near(s1.t, DT, 1e-12), 'got ' + String(s1.t));
}
{
  // The road is elsewhere: the car at checkpoint 0 starts off-road.
  const off = withTrack({centerline: [{x: 600, y: 100}, {x: 900, y: 100}, {x: 900, y: 300}, {x: 600, y: 300}]});
  const s1 = S.step(S.create(off), FULL, DT);
  const want = (220.0 * 0.5 * DT) * Math.pow(0.35, DT);
  ok('off the road: ACCEL is halved and OFFROAD_DRAG applies',
     near(s1.player.speed, want, 1e-4), 'want ' + want.toFixed(6) + ', got ' + String(s1.player.speed));
}
{
  // Exactly roadWidth/2 from a vertical segment: on the road (<=). Half a unit further: off.
  const edgeOn = withTrack({centerline: [{x: 210, y: 100}, {x: 210, y: 700}, {x: 900, y: 700}, {x: 900, y: 100}]});
  const edgeOff = withTrack({centerline: [{x: 210.5, y: 100}, {x: 210.5, y: 700}, {x: 900, y: 700}, {x: 900, y: 100}]});
  const onWant = (220.0 * DT) * Math.pow(0.9, DT), offWant = (110.0 * DT) * Math.pow(0.35, DT);
  ok('the road edge is inclusive (distance == roadWidth/2 is on the road)',
     near(S.step(S.create(edgeOn), FULL, DT).player.speed, onWant, 1e-4));
  ok('just past the road edge is off the road',
     near(S.step(S.create(edgeOff), FULL, DT).player.speed, offWant, 1e-4));
  // The closing segment (last -> first) is part of the road too.
  const closing = withTrack({centerline: [{x: 150, y: 100}, {x: 900, y: 100}, {x: 900, y: 700}, {x: 150, y: 700}]});
  ok('the closing segment of the centerline counts as road',
     near(S.step(S.create(closing), FULL, DT).player.speed, onWant, 1e-4));
}
{
  let s = S.create(cfg());
  for (let i = 0; i < 60; i++) s = S.step(s, FULL, DT);
  const fast = s.player.speed;
  for (let i = 0; i < 60; i++) s = S.step(s, NO_INPUT, DT);
  ok('coasting bleeds speed via drag', s.player.speed < fast && s.player.speed > 0);
  let r = S.create(cfg());
  for (let i = 0; i < 30; i++) r = S.step(r, {throttle: 0, brake: 1, steer: 0, boost: 0}, DT);
  ok('braking from rest never drives speed negative', r.player.speed >= 0);
  const s0 = S.create(cfg()), s1 = S.step(s0, {throttle: 0, brake: 0, steer: 1, boost: 0}, DT);
  ok('steering at zero speed does not rotate the car', near(s1.player.heading, s0.player.heading, 1e-9));
}
{
  // Huge, all-road world so the clamp is reached rather than a wall.
  const big = cfg({opponents: 0, track: {width: 1e9, height: 1e9, roadWidth: 4e9,
    centerline: [{x: 0, y: 0}, {x: 1e9, y: 0}, {x: 1e9, y: 1e9}, {x: 0, y: 1e9}],
    checkpoints: [{x: 5e8, y: 5e8, r: 40}, {x: 5e8 + 1e6, y: 5e8, r: 40}, {x: 5e8, y: 5e8 + 1e6, r: 40}]}});
  let s = S.create(big);
  for (let i = 0; i < 3600; i++) s = S.step(s, FULL, DT);
  ok('speed is clamped to MAX_SPEED and actually reaches it',
     s.player.speed <= 320 + 1e-6 && s.player.speed > 160, 'speed ' + Number(s.player.speed).toFixed(3));
  let peak = 0;
  for (let i = 0; i < 120; i++) { s = S.step(s, BOOST, DT); peak = Math.max(peak, s.player.speed); }
  ok('boost raises the cap: speed exceeds MAX_SPEED but never BOOST_MAX',
     peak > 320 && peak <= 400 + 1e-6, 'peak ' + peak.toFixed(3));
}
{
  // Aimed at the left wall from 30 units away: clamp to x = 0 and stop.
  const wall = cfg({opponents: 0, track: {width: 1000, height: 800, roadWidth: 120,
    centerline: [{x: 0, y: 380}, {x: 200, y: 380}, {x: 200, y: 420}, {x: 0, y: 420}],
    checkpoints: [{x: 30, y: 400, r: 10}, {x: 5, y: 400, r: 1}, {x: 300, y: 400, r: 10}]}});
  let s = S.create(wall);
  for (let i = 0; i < 120; i++) s = S.step(s, FULL, DT);
  ok('walls clamp the position and zero the speed', s.player.x === 0 && s.player.speed === 0,
     'x ' + s.player.x + ' speed ' + s.player.speed);
}

// --- boost: activation, duration, recharge ----------------------------------------------
{
  const s1 = S.step(S.create(cfg()), BOOST, DT);
  const want = (220.0 * 1.8 * DT) * Math.pow(0.9, DT);
  ok('boost: ACCEL x BOOST_MULT on the activation step',
     near(s1.player.speed, want, 1e-4), 'want ' + want.toFixed(6) + ', got ' + String(s1.player.speed));
  ok('boost: charge drops to 0 and boostActive = DURATION - dt',
     s1.player.boostCharge === 0 && near(s1.player.boostActive, 2.0 - DT, 1e-9),
     'charge ' + s1.player.boostCharge + ' active ' + s1.player.boostActive);
  const s2 = S.step(s1, BOOST, DT);
  ok('boost: holding the key does not restart an active boost',
     near(s2.player.boostActive, 2.0 - 2 * DT, 1e-9) && s2.player.boostCharge === 0);
  let s = s1;
  for (let i = 0; i < 124; i++) s = S.step(s, {throttle: 0, brake: 0, steer: 0, boost: 1}, DT);
  ok('boost: expires after BOOST_DURATION', s.player.boostActive === 0, 'active ' + s.player.boostActive);
  ok('boost: cannot re-activate on a partial charge', s.player.boostCharge < 1 && s.player.boostActive === 0,
     'charge ' + s.player.boostCharge);
  for (let i = 0; i < 480; i++) s = S.step(s, NO_INPUT, DT);
  ok('boost: recharges to full in BOOST_RECHARGE seconds',
     s.player.boostCharge >= 0.99 && s.player.boostCharge <= 1, 'charge ' + s.player.boostCharge);
  const s3 = S.step(s, BOOST, DT);
  ok('boost: re-activates once the charge is full', s3.player.boostActive > 0 && s3.player.boostCharge === 0);
}

// --- checkpoints, laps, finish ------------------------------------------------------------
{
  const cps = TRACK.checkpoints;
  const s0 = S.create(cfg({opponents: 0}));
  const cheat = cp(s0); cheat.player.x = cps[3].x; cheat.player.y = cps[3].y;
  const after = S.step(cheat, NO_INPUT, DT);
  ok('out-of-order checkpoint contact is ignored', after.nextCheckpoint === 1 && after.lap === 0);
  const legit = cp(s0); legit.player.x = cps[1].x; legit.player.y = cps[1].y;
  ok('in-order checkpoint contact advances exactly one', S.step(legit, NO_INPUT, DT).nextCheckpoint === 2);
  // Touching the LAST checkpoint wraps nextCheckpoint to 0 and credits the lap (§2.3);
  // the start/finish circle is then simply the next target.
  const circuit = (start, idxs) => {
    let s = start;
    for (const idx of idxs) { s = cp(s); s.player.x = cps[idx].x; s.player.y = cps[idx].y; s = S.step(s, NO_INPUT, DT); }
    return s;
  };
  const s3 = circuit(s0, [1, 2, 3]);
  ok('one full ordered circuit increments lap exactly once', s3.lap === 1 && s3.nextCheckpoint === 0, 'lap ' + s3.lap + ' next ' + s3.nextCheckpoint);
  ok('a completed lap records its time, bestLap and lapStart',
     s3.lapTimes.length === 1 && near(s3.lapTimes[0], 3 * DT, 1e-6) && near(s3.bestLap, s3.lapTimes[0], 1e-9) && near(s3.lapStart, s3.t, 1e-9),
     JSON.stringify({lapTimes: s3.lapTimes, bestLap: s3.bestLap, lapStart: s3.lapStart, t: s3.t}));
  const s = circuit(s3, [0]);
  ok('passing the start/finish circle again does not count a second lap', s.lap === 1 && s.nextCheckpoint === 1 && s.lapTimes.length === 1);
  ok('not finished before `laps` laps', s.finished === false);
  const f = circuit(S.create(cfg({laps: 1, opponents: 0})), [1, 2, 3]);
  ok('finished once `laps` laps are complete', f.finished === true);
  const h = S.hash(f), g = S.step(f, BOOST, DT);
  ok('a finished state is frozen (step returns an identical state)', S.hash(g) === h && near(g.t, f.t, 1e-12));
}

// --- opponents ----------------------------------------------------------------------------
{
  let s = S.create(cfg()), inWorld = true, underCap = true, perm = true;
  const ids = ['player', 'opp0', 'opp1'];
  for (let i = 0; i < 3600; i++) {
    s = S.step(s, NO_INPUT, DT);
    for (const o of s.opponents) {
      if (!(o.x >= 0 && o.x <= 1000 && o.y >= 0 && o.y <= 800)) inWorld = false;
      if (!(o.speed <= 320 + 1e-6)) underCap = false;
    }
    if (i % 30 === 0) {
      const r = s.ranking;
      if (!(Array.isArray(r) && r.length === 3 && ids.every(id => r.includes(id)))) perm = false;
    }
  }
  ok('opponents: the configured count', s.opponents.length === 2);
  ok('opponents stay inside the world', inWorld);
  ok('opponents never exceed MAX_SPEED (no boost for the AI)', underCap);
  const best = Math.max(...s.opponents.map(o => o.lap | 0));
  ok('opponents are competent: a lap within 60 s on the standard track', best >= 1, 'best opponent lap ' + best);
  ok('ranking is always a permutation of the car ids', perm);
  const run = (seed) => { let x = S.create(cfg({seed})); for (let i = 0; i < 600; i++) x = S.step(x, NO_INPUT, DT); return S.hash(x); };
  ok('opponents are seeded: a different seed changes the state', run(12345) !== run(54321));
  const q = S.create(cfg()), q2 = cp(q); q2.opponents[0].x += 1;
  ok('hash changes when an opponent moves', S.hash(q) !== S.hash(q2));
  const l1 = cp(q); l1.lap = 5;
  ok('a car with more laps ranks first (player)', S.step(l1, NO_INPUT, DT).ranking[0] === 'player');
  const l2 = cp(q); l2.opponents[1].lap = 5;
  ok('a car with more laps ranks first (opponent)', S.step(l2, NO_INPUT, DT).ranking[0] === 'opp1');
}

// --- determinism over a long input tape ------------------------------------------------
function tape(seed) {
  let x = seed >>> 0;
  const next = () => (x = (x * 1664525 + 1013904223) >>> 0) / 4294967296;
  const out = [];
  for (let i = 0; i < 600; i++) {
    out.push({throttle: next() > 0.2 ? 1 : 0, brake: next() > 0.9 ? 1 : 0, steer: next() * 2 - 1, boost: next() > 0.97 ? 1 : 0});
  }
  return out;
}
{
  const inputs = tape(99);
  const run = () => { let s = S.create(cfg()); for (const inp of inputs) s = S.step(s, inp, DT); return s; };
  const r1 = run(), r2 = run(), h1 = S.hash(r1), h2 = S.hash(r2);
  ok('600-step tape is bit-identical across two runs', h1 === h2, h1 + ' vs ' + h2);
  ok('hash is a non-trivial string', typeof h1 === 'string' && h1.length >= 8, String(h1));
  let ser = true;
  try { ser = S.hash(cp(r1)) === h1; } catch (e) { ser = false; }
  ok('state is still JSON-serialisable after 600 steps', ser);
  const moved = S.step(r1, FULL, DT);
  ok('hash changes when the state changes', S.hash(moved) !== h1);
}

// --- View.project ---------------------------------------------------------------------------
{
  const cam = {x: 0, y: 0, z: 0, yaw: 0, pitch: 0, fov: 1.0, width: 800, height: 600};
  const c = V.project({x: 100, y: 0, z: 0}, cam);
  ok('View: a point on the viewing axis projects to the viewport centre',
     near(c.sx, 400, 1e-6) && near(c.sy, 300, 1e-6) && c.depth > 0, JSON.stringify(c));
  const n = V.project({x: 100, y: 10, z: 0}, cam), f = V.project({x: 200, y: 10, z: 0}, cam);
  ok('View: perspective foreshortening (farther is deeper and nearer the centre)',
     f.depth > n.depth && Math.abs(f.sx - 400) < Math.abs(n.sx - 400) && Math.abs(f.sx - 400) > 0,
     JSON.stringify({near: n, far: f}));
  ok('View: the right-hand side (yaw + pi/2) lands on the right of the viewport', n.sx > 400, 'sx ' + n.sx);
  const bh = V.project({x: -50, y: 0, z: 0}, cam);
  ok('View: a point behind the camera has depth <= 0', bh.depth <= 0, 'depth ' + bh.depth);
  const c2 = V.project({x: 0, y: 100, z: 0}, Object.assign({}, cam, {yaw: Math.PI / 2}));
  ok('View: yaw follows the heading convention', near(c2.sx, 400, 1e-6) && near(c2.sy, 300, 1e-6) && c2.depth > 0, JSON.stringify(c2));
  const d = V.project({x: 100, y: 0, z: 0}, Object.assign({}, cam, {pitch: -0.3}));
  ok('View: negative pitch looks down (a point ahead on the plane rises on screen)', d.sy < 300 && d.depth > 0, JSON.stringify(d));
  const up = V.project({x: 100, y: 0, z: 20}, cam);
  ok('View: +z is up on screen', up.sy < 300, 'sy ' + up.sy);
}
} catch (e) {
  ok('SimCore/View ran every floor check without throwing', false,
     String(e && e.stack ? e.stack.split('\n').slice(0, 2).join(' | ') : e).slice(0, 300));
}

console.log(JSON.stringify(results));
"""

HARNESSES = {
    "racing-v1": HARNESS_PRELUDE_JS + HARNESS_V1_JS,
    "racing-v2": HARNESS_PRELUDE_JS + HARNESS_V2_JS,
}


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
    ap.add_argument("--spec", default="racing-v1", choices=sorted(HARNESSES),
                    help="which spec's floor to apply (default: racing-v1)")
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
            js.write_text(HARNESSES[args.spec], encoding="utf-8")
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
