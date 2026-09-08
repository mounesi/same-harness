# racing-v1 — top-down racing game

Build a playable top-down racing game as **one self-contained HTML file**.

This spec is the whole brief. It is deliberately precise about the simulation and
deliberately quiet about everything else: the physics is graded by machine, the look and
the feel are judged by people. Where this document does not constrain you, you are free —
and that freedom is the part being evaluated.

---

## 1. Deliverable

A single file, `game.html`. It must:

- open and play in a modern browser by double-clicking it — **no build step, no server,
  no network access at runtime, no external assets, no CDN**
- be under 256 KB
- render to a `<canvas>` and be playable with the keyboard

Everything — markup, styles, code — lives in that one file. Embed any image or audio you
want as a `data:` URI, or generate it procedurally.

## 2. The simulation core (this is the graded part)

The file MUST define a global `SimCore` with exactly this shape. An automated check drives
it headlessly, with no DOM, so **`SimCore` must not touch `window`, `document`, `canvas`,
timers, or `Math.random` at module scope or inside any method below.**

```js
globalThis.SimCore = {
  create(config),                  // -> state   (pure; same config => same state)
  step(state, input, dt),          // -> state   (pure; returns NEW state, never mutates)
  hash(state),                     // -> string  (stable digest of the full state)
};
```

### 2.1 `create(config)`

`config` is `{ seed, track }`.

- `seed` — a 32-bit unsigned integer. **All randomness must derive from it.** A call to
  `Math.random()` anywhere in `SimCore` is a defect.
- `track` — `{ width, height, checkpoints }`. `checkpoints` is an ordered array of at
  least 3 `{ x, y, r }` circles. The first is the start/finish line.

Returns a `state` object. `state` must be JSON-serialisable — no functions, no `Map`,
no `Set`, no cyclic references.

The car starts at `checkpoints[0]` with `speed = 0`, heading toward `checkpoints[1]`.

### 2.2 `step(state, input, dt)`

`input` is `{ throttle, brake, steer }`:

| field | range | meaning |
|---|---|---|
| `throttle` | `0..1` | forward force |
| `brake` | `0..1` | deceleration |
| `steer` | `-1..1` | −1 full left, +1 full right |

`dt` is seconds, always `1/60` in grading.

Required physics, exactly:

```
ACCEL     = 220.0     units/s^2 at throttle = 1
BRAKE     = 400.0     units/s^2 at brake = 1
DRAG      = 0.9       multiplicative per second: speed *= DRAG ** dt
TURN_RATE = 2.6       radians/s at |steer| = 1, scaled by min(1, speed / 60)
MAX_SPEED = 320.0     units/s
```

Per step, in this order:

1. `speed += (ACCEL * throttle - BRAKE * brake) * dt`
2. `speed *= DRAG ** dt`
3. clamp `speed` to `0 .. MAX_SPEED`
4. `heading += steer * TURN_RATE * min(1, speed / 60) * dt`
5. `x += cos(heading) * speed * dt`, `y += sin(heading) * speed * dt`
6. resolve checkpoints, then laps (§2.3)

The car must not leave the track rectangle: clamp `x` to `0..width`, `y` to `0..height`,
and set `speed = 0` on a wall contact.

### 2.3 Checkpoints and laps — ordered, and this is the anti-cheat

`state.nextCheckpoint` starts at `1`. When the car's centre is within `r` of
`checkpoints[nextCheckpoint]`, advance it (wrapping to `0`).

**Checkpoints only count in order.** Touching checkpoint 3 while 2 is still pending does
nothing at all. A lap increments only when `nextCheckpoint` wraps from the last back to
`0`, which is what makes cutting the course impossible rather than merely discouraged.

`state.lap` starts at `0`.

### 2.4 `hash(state)`

A stable string digest of the entire state. Requirements:

- identical states → identical string
- any change to any numeric field → different string
- **independent of key insertion order** (sort keys; do not rely on `JSON.stringify`
  ordering)
- floats rounded to 6 decimal places before hashing, so that arithmetic that is
  mathematically equal but differs in the last bit still agrees

## 3. Determinism (graded, and the hardest requirement here)

Given the same `seed` and the same sequence of inputs, `SimCore` must produce a
**bit-identical** state after any number of steps. That means:

- no `Math.random()`, no `Date.now()`, no `performance.now()` inside `SimCore`
- no iteration over anything whose order is not defined
- no dependence on wall-clock time or frame rate — `dt` is a parameter, never measured

If you add AI opponents, they are part of the state and bound by all of the above.

## 4. The game (judged by people, not by machine)

Beyond the above you decide everything: how it looks, how it feels to drive, what the
track looks like, whether there are opponents, a HUD, a lap timer, sound, a menu, a
minimap, damage, drifting, weather.

The rendering layer may use `Math.random`, timers, and anything else it likes. **Only
`SimCore` is constrained** — keep the split clean and both halves get easier.

Controls must be discoverable without instructions. Arrow keys or WASD are expected.

## 5. What is checked automatically

The automated floor is pass/fail and only asks whether you built a working thing:

1. `game.html` exists, is under 256 KB, and references no external URL
2. `SimCore` loads in a bare JS context — no DOM, no browser globals
3. `create()` is pure: same config twice → identical `hash`
4. `step()` matches the specified physics on hand-computed cases
5. `step()` does not mutate its argument
6. checkpoints only advance in order; out-of-order contact is ignored
7. a lap increments exactly once per full ordered circuit
8. determinism: the same 600-step input tape produces the same final hash, twice
9. the state stays JSON-serialisable throughout

Failing any of these means the submission does not enter human judging at all. Passing
them says nothing about whether the game is any good — that is the other half, and it is
judged blind by people playing it.
