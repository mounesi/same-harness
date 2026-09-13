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

# racing-v2 (City Run) — see specs/racing-v2.md §6. Every constant below is a spec constant.
HARNESS_V2_JS = r"""
const S = globalThis.SimCore, V = globalThis.View;
ok('SimCore global is defined', !!S);
ok('View global is defined', !!V);
if (!S || !V) { console.log(JSON.stringify(results)); process.exit(0); }
const fns = (o, names) => names.every(n => typeof o[n] === 'function');
ok('SimCore has presets/create/step/hash', fns(S, ['presets', 'create', 'step', 'hash']));
ok('View has project', fns(V, ['project']));
if (!fns(S, ['presets', 'create', 'step', 'hash']) || !fns(V, ['project'])) {
  console.log(JSON.stringify(results)); process.exit(0);
}

const TRACK = {width: 1000, height: 800, roadWidth: 120,
  roads: [[{x: 150, y: 150}, {x: 850, y: 150}, {x: 850, y: 650}, {x: 150, y: 650}, {x: 150, y: 150}],
          [{x: 500, y: 150}, {x: 500, y: 650}], [{x: 150, y: 400}, {x: 850, y: 400}]],
  spawn: {x: 150, y: 400, heading: 0},
  pickups: [[500, 400], [500, 150], [850, 400], [500, 650], [150, 150], [850, 150], [850, 650], [150, 650]]
    .map(q => ({x: q[0], y: q[1], r: 30, value: 10}))};
const CARS = [{id: 'balanced', accel: 1.00, turn: 1.00, mass: 1.0},
              {id: 'sprint', accel: 1.15, turn: 0.90, mass: 0.8},
              {id: 'heavy', accel: 0.90, turn: 0.85, mass: 1.3}];
const CONFIG = {seed: 12345, car: 0, track: TRACK, level: 0, time: 60, opponents: 2, timeOfDay: 'day'};
const LEVELS = [
  {level: 1, time: 60, target: 40, opponents: 1, timeOfDay: 'day'},
  {level: 2, time: 60, target: 60, opponents: 2, timeOfDay: 'day'},
  {level: 3, time: 60, target: 80, opponents: 2, timeOfDay: 'dusk'},
  {level: 4, time: 90, target: 120, opponents: 3, timeOfDay: 'night'},
  {level: 5, time: 90, target: 150, opponents: 3, timeOfDay: 'dawn'},
  {level: 6, time: 90, target: 180, opponents: 4, timeOfDay: 'day'},
  {level: 7, time: 120, target: 240, opponents: 4, timeOfDay: 'dusk'},
  {level: 8, time: 120, target: 300, opponents: 5, timeOfDay: 'night'},
];
const lvl = (n) => ({seed: 1, car: 0, track: JSON.parse(JSON.stringify(TRACK)), level: n});
const cfg = (over) => JSON.parse(JSON.stringify(Object.assign({}, CONFIG, over || {})));
const withTrack = (over) => cfg({track: Object.assign({}, TRACK, over)});
const NO_INPUT = {throttle: 0, brake: 0, steer: 0, boost: 0};
const FULL = {throttle: 1, brake: 0, steer: 0, boost: 0};
const BOOST = {throttle: 1, brake: 0, steer: 0, boost: 1};
const DT = 1 / 60;
const cp = (s) => JSON.parse(JSON.stringify(s));
const onWant = (accel) => (220.0 * accel * DT) * Math.pow(0.9, DT);

// The harness's own road test, straight from §2.1, to validate preset maps.
function segD2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  let t = l2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0; t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx, cy = ay + t * dy; return (px - cx) ** 2 + (py - cy) ** 2;
}
function onRoad(track, x, y) {
  let best = Infinity;
  for (const road of track.roads) for (let i = 0; i + 1 < road.length; i++)
    best = Math.min(best, segD2(x, y, road[i].x, road[i].y, road[i + 1].x, road[i + 1].y));
  return best <= (track.roadWidth / 2) ** 2;
}

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
  S.presets();
  const s = S.create(cfg());
  V.project({x: 1, y: 2, z: 0}, {x: 0, y: 0, z: 10, yaw: 0.3, pitch: -0.3, fov: 1, width: 800, height: 600});
  if (typeof V.scenery === 'function') V.scenery(JSON.parse(JSON.stringify(TRACK)), 12345);
  return S.hash(S.step(s, BOOST, DT));
});
ok('SimCore and View touch no browser globals or RNG', !pure.error,
   pure.error ? String(pure.error.message).slice(0, 160) : '');

// --- presets(): the start screen's options ----------------------------------------------
{
  const P = S.presets() || {};
  ok('presets() reports the cars table exactly', JSON.stringify(P.cars) === JSON.stringify(CARS),
     JSON.stringify(P.cars).slice(0, 200));
  ok('presets() reports times [30, 60, 120]', JSON.stringify(P.times) === '[30,60,120]', JSON.stringify(P.times));
  ok('presets() reports the level table exactly (8 rows)', JSON.stringify(P.levels) === JSON.stringify(LEVELS),
     JSON.stringify(P.levels).slice(0, 200));
  ok('presets() offers at least two maps', Array.isArray(P.tracks) && P.tracks.length >= 2,
     'tracks: ' + (Array.isArray(P.tracks) ? P.tracks.length : typeof P.tracks));
  let valid = true, why = '';
  for (const [k, t] of (Array.isArray(P.tracks) ? P.tracks : []).entries()) {
    const tr = t && t.track;
    if (!tr || !t.name) { valid = false; why = 'entry ' + k + ' has no name/track'; break; }
    if (!(tr.width > 0 && tr.height > 0 && tr.roadWidth > 0 && Array.isArray(tr.roads) && tr.roads.length >= 1 &&
          tr.roads.every(r => Array.isArray(r) && r.length >= 2))) { valid = false; why = t.name + ': size/streets'; break; }
    if (!(Array.isArray(tr.pickups) && tr.pickups.length >= 6 &&
          tr.pickups.every(q => q.r > 0 && q.value >= 1 && onRoad(tr, q.x, q.y)))) { valid = false; why = t.name + ': needs >= 6 pickups, all on the road, r > 0, value >= 1'; break; }
    if (!(tr.spawn && onRoad(tr, tr.spawn.x, tr.spawn.y))) { valid = false; why = t.name + ': spawn is off the road'; break; }
  }
  ok('every preset map is valid (streets, >= 6 pickups on the road, spawn on the road)', valid, why);
  let runs = true, rwhy = '';
  try {
    for (const t of (Array.isArray(P.tracks) ? P.tracks : [])) for (let c = 0; c < 3; c++) for (const tm of [30, 60, 120]) {
      let s = S.create({seed: 1, car: c, track: JSON.parse(JSON.stringify(t.track)), level: 0, time: tm, opponents: 2, timeOfDay: 'day'});
      for (let i = 0; i < 60; i++) s = S.step(s, FULL, DT);
      if (typeof s.score !== 'number' || s.timeLimit !== tm || s.player.car !== c) throw new Error('bad free-run state on ' + t.name);
    }
    for (const t of (Array.isArray(P.tracks) ? P.tracks : [])) for (let n = 1; n <= 8; n++) {
      let s = S.create({seed: 1, car: 1, track: JSON.parse(JSON.stringify(t.track)), level: n});
      for (let i = 0; i < 60; i++) s = S.step(s, FULL, DT);
      if (s.level !== n || s.timeLimit !== LEVELS[n - 1].time) throw new Error('bad level-' + n + ' state on ' + t.name);
    }
  } catch (e) { runs = false; rwhy = String(e).slice(0, 160); }
  ok('a run can be created and stepped on every preset map, car, free-run time and level', runs, rwhy);
  ok('presets() is constant', JSON.stringify(P) === JSON.stringify(S.presets()));
}

// --- create(): determinism, shape, spawn, serialisability, no mutation ------------------
const a = S.create(cfg()), b = S.create(cfg());
ok('create() is deterministic for one config', S.hash(a) === S.hash(b));
{
  const s = a, p = s.player;
  ok('initial state carries the required fields',
     s.t === 0 && s.timeLimit === 60 && s.timeLeft === 60 && s.finished === false && s.score === 0 &&
     s.level === 0 && s.target === null && s.timeOfDay === 'day' && s.outcome === null && s.campaignWon === false &&
     s.crashes === 0 && s.round === 0 && !!p && p.speed === 0 && p.car === 0 && p.boostCharge === 1 &&
     p.boostActive === 0 && Array.isArray(s.opponents) && s.opponents.length === 2 &&
     Array.isArray(s.pickups) && s.pickups.length === 8 && s.pickups.every(q => q.taken === false && q.value === 10 && q.r === 30) &&
     Array.isArray(s.ranking),
     JSON.stringify({t: s.t, timeLimit: s.timeLimit, timeLeft: s.timeLeft, finished: s.finished, score: s.score,
                     level: s.level, target: s.target, timeOfDay: s.timeOfDay, outcome: s.outcome, campaignWon: s.campaignWon,
                     crashes: s.crashes, round: s.round, player: p, opponents: (s.opponents || []).length,
                     pickups: (s.pickups || []).length}).slice(0, 260));
  ok('the player spawns at spawn, facing spawn.heading', near(p.x, 150) && near(p.y, 400) && near(p.heading, 0, 1e-9));
  ok('opponent i spawns 40*(i+1) units ahead along the spawn heading, facing the same way',
     s.opponents.every((o, i) => near(o.x, 150 + 40 * (i + 1), 1e-6) && near(o.y, 400, 1e-6) && near(o.heading, 0, 1e-9) &&
                                 o.speed === 0 && o.score === 0 && o.crashes === 0 && [0, 1, 2].includes(o.car)),
     JSON.stringify(s.opponents.map(o => [o.x, o.y, o.heading, o.car])));
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
  ok('on the road: speed after one full-throttle step matches the spec (balanced)',
     near(s1.player.speed, onWant(1), 1e-4), 'want ' + onWant(1).toFixed(6) + ', got ' + String(s1.player.speed));
  ok('t advances by dt each step', near(s1.t, DT, 1e-12), 'got ' + String(s1.t));
  const s2 = S.step(S.create(cfg({car: 1})), FULL, DT);
  ok('the sprint car accelerates 1.15x', near(s2.player.speed, onWant(1.15), 1e-4), 'want ' + onWant(1.15).toFixed(6) + ', got ' + String(s2.player.speed));
  const s3 = S.step(S.create(cfg({car: 2})), FULL, DT);
  ok('the heavy car accelerates 0.9x', near(s3.player.speed, onWant(0.9), 1e-4), 'want ' + onWant(0.9).toFixed(6) + ', got ' + String(s3.player.speed));
  // turn multiplier: at speed >= 60 the heading changes by steer * TURN_RATE * turn * dt
  const turnOf = (car) => { const s = cp(S.create(cfg({car, opponents: 0}))); s.player.speed = 100; const n = S.step(s, {throttle: 0, brake: 0, steer: 1, boost: 0}, DT); return n.player.heading - s.player.heading; };
  ok('the heavy car turns 0.85x, the sprint car 0.9x',
     near(turnOf(0), 2.6 * DT, 1e-9) && near(turnOf(1), 2.6 * 0.9 * DT, 1e-9) && near(turnOf(2), 2.6 * 0.85 * DT, 1e-9),
     JSON.stringify([turnOf(0), turnOf(1), turnOf(2)]));
}
{
  // Streets elsewhere: the car spawns off the road.
  const off = withTrack({roads: [[{x: 600, y: 100}, {x: 900, y: 100}, {x: 900, y: 300}]]});
  const s1 = S.step(S.create(off), FULL, DT);
  const want = (220.0 * 0.5 * DT) * Math.pow(0.35, DT);
  ok('off the road: acceleration is halved and OFFROAD_DRAG applies',
     near(s1.player.speed, want, 1e-4), 'want ' + want.toFixed(6) + ', got ' + String(s1.player.speed));
  // Exactly roadWidth/2 from a street: on the road (<=). Half a unit further: off.
  const edgeOn = withTrack({roads: [[{x: 210, y: 100}, {x: 210, y: 700}]]});
  const edgeOff = withTrack({roads: [[{x: 210.5, y: 100}, {x: 210.5, y: 700}]]});
  const offWant = (110.0 * DT) * Math.pow(0.35, DT);
  ok('the road edge is inclusive (distance == roadWidth/2 is on the road)',
     near(S.step(S.create(edgeOn), FULL, DT).player.speed, onWant(1), 1e-4));
  ok('just past the road edge is off the road',
     near(S.step(S.create(edgeOff), FULL, DT).player.speed, offWant, 1e-4));
  // Street ends are rounded: 50 past an endpoint is still road, 65 is not.
  const endOn = withTrack({roads: [[{x: 200, y: 400}, {x: 100, y: 400}]], spawn: {x: 250, y: 400, heading: 0}});
  const endOff = withTrack({roads: [[{x: 200, y: 400}, {x: 100, y: 400}]], spawn: {x: 265, y: 400, heading: 0}});
  ok('street ends are rounded (within roadWidth/2 of an endpoint is road)',
     near(S.step(S.create(endOn), FULL, DT).player.speed, onWant(1), 1e-4) &&
     near(S.step(S.create(endOff), FULL, DT).player.speed, offWant, 1e-4));
}
{
  let s = S.create(cfg({opponents: 0}));
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
    roads: [[{x: 0, y: 5e8}, {x: 1e9, y: 5e8}]], spawn: {x: 5e8, y: 5e8, heading: 0},
    pickups: [1, 2, 3, 4, 5, 6].map(i => ({x: 5e8 + i * 1e7, y: 5e8, r: 30, value: 10}))}});
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
  // Facing the left wall from 30 units away: clamp to x = 0 and stop.
  const wall = withTrack({roads: [[{x: 0, y: 400}, {x: 300, y: 400}]], spawn: {x: 30, y: 400, heading: Math.PI}});
  let s = S.create(cfg({opponents: 0, track: wall.track}));
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

// --- collisions ----------------------------------------------------------------------------
{
  const base = S.create(cfg({opponents: 1}));
  const place = (px, py, ox, oy) => {
    const s = cp(base);
    s.player.x = px; s.player.y = py; s.player.heading = 0; s.player.speed = 100;
    s.opponents[0].x = ox; s.opponents[0].y = oy; s.opponents[0].heading = 0; s.opponents[0].speed = 0;
    return s;
  };
  const hit = S.step(place(400, 400, 412, 400), NO_INPUT, DT);      // will overlap after moving
  const freeP = S.step(place(400, 400, 900, 620), NO_INPUT, DT);    // the player, undisturbed
  const freeO = S.step(place(400, 700, 412, 400), NO_INPUT, DT);    // the opponent, undisturbed
  const mo = CARS[base.opponents[0].car].mass, mp = CARS[0].mass;
  const fp = 1 - 0.6 * mo / (mp + mo), fo = 1 - 0.6 * mp / (mp + mo);
  const d = Math.hypot(hit.player.x - hit.opponents[0].x, hit.player.y - hit.opponents[0].y);
  ok('collision: overlapping cars are pushed to exactly 2 * CAR_RADIUS apart', near(d, 20, 1e-6), 'distance ' + d);
  ok('collision: both cars slow by the mass rule',
     near(hit.player.speed, freeP.player.speed * fp, 1e-6) && near(hit.opponents[0].speed, freeO.opponents[0].speed * fo, 1e-6),
     JSON.stringify({player: [hit.player.speed, freeP.player.speed * fp], opponent: [hit.opponents[0].speed, freeO.opponents[0].speed * fo], mo}));
  ok('collision: both cars lose CRASH_PENALTY and count a crash',
     hit.score === -25 && hit.crashes === 1 && hit.opponents[0].score === -25 && hit.opponents[0].crashes === 1,
     JSON.stringify({score: hit.score, crashes: hit.crashes, opp: [hit.opponents[0].score, hit.opponents[0].crashes]}));
  ok('no collision when the cars are apart', freeP.score === 0 && freeP.crashes === 0);
  const rest = cp(hit); rest.player.speed = 0;
  const again = S.step(rest, NO_INPUT, DT);
  ok('a touching pair does not crash again unless it closes in', again.crashes === 1 && again.score === -25,
     JSON.stringify({crashes: again.crashes, score: again.score}));
}

// --- pickups ----------------------------------------------------------------------------------
{
  const pk = TRACK.pickups;
  let s = cp(S.create(cfg({opponents: 0})));
  s.player.x = pk[0].x; s.player.y = pk[0].y; s = S.step(s, NO_INPUT, DT);
  ok('reaching a pickup scores its value and marks it taken', s.score === 10 && s.pickups[0].taken === true,
     JSON.stringify({score: s.score, taken: s.pickups[0].taken}));
  s = S.step(s, NO_INPUT, DT);
  ok('a taken pickup does not score again', s.score === 10, 'score ' + s.score);
  for (let i = 1; i < pk.length; i++) { s = cp(s); s.player.x = pk[i].x; s.player.y = pk[i].y; s = S.step(s, NO_INPUT, DT); }
  ok('clearing the set resets every pickup and increments round',
     s.score === 80 && s.round === 1 && s.pickups.every(q => q.taken === false),
     JSON.stringify({score: s.score, round: s.round, taken: s.pickups.map(q => q.taken)}));
  s = S.step(s, NO_INPUT, DT);
  ok('after the reset the pickup under the car scores again', s.score === 90, 'score ' + s.score);
  const near1 = cp(S.create(cfg({opponents: 0}))); near1.player.x = pk[0].x + 29; near1.player.y = pk[0].y;
  const far1 = cp(S.create(cfg({opponents: 0}))); far1.player.x = pk[0].x + 31; far1.player.y = pk[0].y;
  ok('a pickup counts within r and not beyond it',
     S.step(near1, NO_INPUT, DT).score === 10 && S.step(far1, NO_INPUT, DT).score === 0);
}

// --- the timer ------------------------------------------------------------------------------
{
  let s = S.create(cfg({time: 30, opponents: 0}));
  for (let i = 0; i < 1798; i++) s = S.step(s, NO_INPUT, DT);
  ok('not finished before the time limit', s.finished === false && s.timeLeft > 0, JSON.stringify({t: s.t, timeLeft: s.timeLeft}));
  for (let i = 0; i < 4; i++) s = S.step(s, NO_INPUT, DT);
  ok('free run: finished and outcome "done" once t reaches the time limit, with timeLeft 0',
     s.finished === true && s.timeLeft === 0 && s.outcome === 'done' && s.campaignWon === false,
     JSON.stringify({t: s.t, timeLeft: s.timeLeft, finished: s.finished, outcome: s.outcome}));
  const h = S.hash(s), g = S.step(s, BOOST, DT);
  ok('a finished state is frozen (step returns an identical state)', S.hash(g) === h && near(g.t, s.t, 1e-12));
  ok('free run: timeOfDay comes from the config and defaults to day',
     S.create(cfg({timeOfDay: 'night'})).timeOfDay === 'night' && S.create(cfg({timeOfDay: undefined})).timeOfDay === 'day');
}

// --- campaign levels -------------------------------------------------------------------------
{
  let rows = true, why = '';
  for (let n = 1; n <= 8; n++) {
    const s = S.create(lvl(n)), r = LEVELS[n - 1];
    if (!(s.level === n && s.timeLimit === r.time && s.timeLeft === r.time && s.target === r.target &&
          s.opponents.length === r.opponents && s.timeOfDay === r.timeOfDay && s.outcome === null && s.campaignWon === false)) {
      rows = false; why = 'level ' + n + ': ' + JSON.stringify({timeLimit: s.timeLimit, target: s.target, opponents: (s.opponents || []).length, timeOfDay: s.timeOfDay}); break;
    }
  }
  ok('each campaign level takes time, target, opponents and time of day from the table', rows, why);
  const ign = S.create(Object.assign(lvl(3), {time: 30, opponents: 0, timeOfDay: 'night'}));
  ok('a campaign level ignores time/opponents/timeOfDay in the config',
     ign.timeLimit === 60 && ign.opponents.length === 2 && ign.timeOfDay === 'dusk');
  const pk = TRACK.pickups;
  let s1 = S.create(lvl(1));
  for (let i = 0; i < 4; i++) { s1 = cp(s1); s1.player.x = pk[i].x; s1.player.y = pk[i].y; s1 = S.step(s1, NO_INPUT, DT); }
  ok('level 1 is won the moment score reaches the target (40)',
     s1.score >= 40 && s1.outcome === 'won' && s1.finished === true && s1.campaignWon === false,
     JSON.stringify({score: s1.score, outcome: s1.outcome, finished: s1.finished, campaignWon: s1.campaignWon}));
  const h1 = S.hash(s1);
  ok('a won level is frozen', S.hash(S.step(s1, FULL, DT)) === h1);
  let s3 = cp(S.create(lvl(1))); s3.player.x = pk[0].x; s3.player.y = pk[0].y; s3 = S.step(s3, NO_INPUT, DT);
  ok('below the target the level continues', s3.score === 10 && s3.outcome === null && s3.finished === false);
  let s8 = cp(S.create(lvl(8))); s8.score = 299; s8.player.x = pk[0].x; s8.player.y = pk[0].y; s8 = S.step(s8, NO_INPUT, DT);
  ok('winning level 8 sets campaignWon', s8.outcome === 'won' && s8.campaignWon === true, JSON.stringify({score: s8.score, outcome: s8.outcome, campaignWon: s8.campaignWon}));
  let tie = cp(S.create(lvl(1))); tie.t = 59.999; tie.score = 39; tie.player.x = pk[0].x; tie.player.y = pk[0].y; tie = S.step(tie, NO_INPUT, DT);
  ok('reaching the target on the step the clock runs out counts as a win', tie.outcome === 'won', JSON.stringify({t: tie.t, score: tie.score, outcome: tie.outcome}));
  let lost = S.create(lvl(1));
  for (let i = 0; i < 3700; i++) lost = S.step(lost, NO_INPUT, DT);
  ok('a level is lost when the clock runs out below the target',
     lost.outcome === 'lost' && lost.finished === true && lost.campaignWon === false && lost.score < 40,
     JSON.stringify({t: lost.t, score: lost.score, outcome: lost.outcome}));
}

// --- scenery ------------------------------------------------------------------------------------
{
  const maps = [{name: 'standard', track: TRACK}].concat(Array.isArray((S.presets() || {}).tracks) ? S.presets().tracks : []);
  ok('View has scenery', typeof V.scenery === 'function');
  if (typeof V.scenery === 'function') {
    let allOk = true, why = '';
    for (const m of maps) {
      const tr = JSON.parse(JSON.stringify(m.track));
      const sc = V.scenery(tr, 12345), sc2 = V.scenery(JSON.parse(JSON.stringify(m.track)), 12345);
      if (!Array.isArray(sc)) { allOk = false; why = m.name + ': scenery is not an array'; break; }
      if (JSON.stringify(sc) !== JSON.stringify(sc2)) { allOk = false; why = m.name + ': scenery is not deterministic'; break; }
      const b = sc.filter(o => o && o.type === 'building');
      if (b.length < 20) { allOk = false; why = m.name + ': ' + b.length + ' buildings (need >= 20)'; break; }
      const bad = b.find(o => !(o.w > 0 && o.d > 0 && o.h > 0) ||
        [[o.x, o.y], [o.x - o.w / 2, o.y - o.d / 2], [o.x + o.w / 2, o.y - o.d / 2], [o.x - o.w / 2, o.y + o.d / 2], [o.x + o.w / 2, o.y + o.d / 2]]
          .some(([x, y]) => onRoad(tr, x, y) || x < 0 || x > tr.width || y < 0 || y > tr.height));
      if (bad) { allOk = false; why = m.name + ': a building stands on the road or outside the world: ' + JSON.stringify(bad).slice(0, 120); break; }
      if (new Set(b.map(o => o.h)).size < 3) { allOk = false; why = m.name + ': fewer than 3 distinct building heights'; break; }
    }
    ok('scenery: >= 20 buildings of >= 3 heights, every footprint off the road and inside the world, on every map', allOk, why);
  }
}

// --- opponents --------------------------------------------------------------------------------
{
  let s = S.create(cfg()), inWorld = true, underCap = true, perm = true;
  const ids = ['player', 'opp0', 'opp1'];
  for (let i = 0; i < 1800; i++) {
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
  ok('opponents are competent: a pickup within 30 s on the standard map',
     s.round > 0 || s.pickups.some(q => q.taken), JSON.stringify({round: s.round, taken: s.pickups.filter(q => q.taken).length, scores: s.opponents.map(o => o.score)}));
  ok('ranking is always a permutation of the car ids', perm);
  const run = (seed) => { let x = S.create(cfg({seed})); for (let i = 0; i < 600; i++) x = S.step(x, NO_INPUT, DT); return S.hash(x); };
  ok('opponents are seeded: a different seed changes the state', run(12345) !== run(54321));
  const q = S.create(cfg()), q2 = cp(q); q2.opponents[0].x += 1;
  ok('hash changes when an opponent moves', S.hash(q) !== S.hash(q2));
  const q3 = cp(q); q3.pickups[2].taken = true;
  ok('hash changes when a pickup is taken', S.hash(q) !== S.hash(q3));
}

// --- ranking ------------------------------------------------------------------------------------
{
  const q = S.create(cfg());
  ok('ranking: on a full tie the player is first, then opponents by index',
     JSON.stringify(S.step(q, NO_INPUT, DT).ranking) === JSON.stringify(['player', 'opp0', 'opp1']), JSON.stringify(q.ranking));
  const r1 = cp(q); r1.opponents[1].score = 50;
  ok('ranking: the highest score ranks first', S.step(r1, NO_INPUT, DT).ranking[0] === 'opp1');
  const r2 = cp(q); r2.opponents[0].score = 50; r2.opponents[1].score = 50;
  ok('ranking: tied opponents keep index order', JSON.stringify(S.step(r2, NO_INPUT, DT).ranking) === JSON.stringify(['opp0', 'opp1', 'player']));
}

// --- determinism over a long input tape --------------------------------------------------
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
  ok('View: negative pitch looks down (a point ahead on the ground rises on screen)', d.sy < 300 && d.depth > 0, JSON.stringify(d));
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
    if args.spec == "racing-v2":
        # §5/§6.16: the music is procedural through the Web Audio API. The machine can only
        # see that the API is used at all; whether the music is any good is for the judges.
        checks.append({"name": "music: the file uses the Web Audio API (AudioContext)",
                       "ok": "AudioContext" in html, "detail": ""})

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
