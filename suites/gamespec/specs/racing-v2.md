# racing-v2 — a 3D race: laps, surfaces, opponents, boost

Build a playable racing game as **one self-contained HTML file**, presented in **3D**.

This spec is the whole brief. It is precise about the simulation and the projection, and
quiet about everything else: the physics and the camera maths are graded by machine, the
look and the feel are judged by people. Where this document does not constrain you, you
are free — and that freedom is the part being evaluated.

---

## 1. Deliverable

A single file, `game.html`. It must:

- open and play in a modern browser by double-clicking it — **no build step, no server,
  no network access at runtime, no external assets, no CDN, no external libraries**
- be under 256 KB
- render a **3D perspective view** (a chase camera behind the player's car, or a cockpit
  view) to a `<canvas>` — WebGL or your own software projection onto a 2D canvas are
  both fine; a top-down 2D view does not satisfy this spec
- be playable with the keyboard

Everything — markup, styles, code — lives in that one file. Embed any image or audio you
want as a `data:` URI, or generate it procedurally.

## 2. The simulation core (graded)

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

`config` is `{ seed, laps, opponents, track }`.

- `seed` — a 32-bit unsigned integer. **All randomness must derive from it.** A call to
  `Math.random()` anywhere in `SimCore` is a defect.
- `laps` — laps to finish the race, ≥ 1.
- `opponents` — number of computer-driven cars, ≥ 0.
- `track` — `{ width, height, roadWidth, centerline, checkpoints }`:
  - `width`, `height` — the world rectangle. Nothing may leave it.
  - `centerline` — an ordered array of at least 4 `{ x, y }` points describing a **closed
    loop** (the last point connects back to the first). The road is every point whose
    distance to that polyline is at most `roadWidth / 2`. Everything else is off-road.
  - `checkpoints` — an ordered array of at least 3 `{ x, y, r }` circles. The first is the
    start/finish line.

Returns a `state` object. `state` must be JSON-serialisable — no functions, no `Map`,
no `Set`, no cyclic references — and must contain **at least** these fields (add whatever
else you need):

```
t              seconds elapsed, starts 0
laps           copied from config
finished       false until the player completes `laps` laps
lap            player's completed laps, starts 0
nextCheckpoint player's next checkpoint index, starts 1
lapStart       t at which the current lap began, starts 0
lapTimes       array of the player's completed lap durations, starts []
bestLap        the smallest entry of lapTimes, or null
player         { x, y, heading, speed, boostCharge, boostActive }
opponents      array of `config.opponents` cars: { x, y, heading, speed, lap, nextCheckpoint }
ranking        array of ids, best first: "player", "opp0", "opp1", …
```

The player and every opponent start at `checkpoints[0]` with `speed = 0`, heading toward
`checkpoints[1]`. Cars overlap freely — **there are no car-to-car collisions in this spec.**
`boostCharge` starts at `1`, `boostActive` at `0`.

### 2.2 `step(state, input, dt)`

`input` is `{ throttle, brake, steer, boost }`:

| field | range | meaning |
|---|---|---|
| `throttle` | `0..1` | forward force |
| `brake` | `0..1` | deceleration |
| `steer` | `-1..1` | −1 full left, +1 full right |
| `boost` | `0` or `1` | request the boost |

`dt` is seconds, always `1/60` in grading.

**If `state.finished` is true, `step` returns a state identical to its input** (same hash),
whatever the input.

Required constants, exactly:

```
ACCEL          = 220.0   units/s^2 at throttle = 1, on the road
BRAKE          = 400.0   units/s^2 at brake = 1
DRAG           = 0.9     multiplicative per second on the road: speed *= DRAG ** dt
TURN_RATE      = 2.6     radians/s at |steer| = 1, scaled by min(1, speed / 60)
MAX_SPEED      = 320.0   units/s

OFFROAD_ACCEL  = 0.5     ACCEL is multiplied by this off the road
OFFROAD_DRAG   = 0.35    replaces DRAG off the road

BOOST_MULT     = 1.8     ACCEL is multiplied by this while the boost is active
BOOST_MAX      = 400.0   replaces MAX_SPEED while the boost is active
BOOST_DURATION = 2.0     seconds a boost lasts
BOOST_RECHARGE = 8.0     seconds from empty to full charge
```

Per step, **for the player**, in this order:

1. `t += dt`
2. `onRoad` = the distance from the player's position **before this step** to the
   centerline polyline is ≤ `roadWidth / 2` (point-to-closed-polyline distance: the
   minimum over every segment, including the closing segment)
3. boost: if `input.boost` is 1 **and** `boostActive == 0` **and** `boostCharge >= 1`,
   set `boostActive = BOOST_DURATION` and `boostCharge = 0`. Then `boosting = boostActive > 0`.
4. `accel = ACCEL * (onRoad ? 1 : OFFROAD_ACCEL) * (boosting ? BOOST_MULT : 1)`,
   `drag = onRoad ? DRAG : OFFROAD_DRAG`, `maxSpeed = boosting ? BOOST_MAX : MAX_SPEED`
5. `speed += (accel * throttle - BRAKE * brake) * dt`
6. `speed *= drag ** dt`
7. clamp `speed` to `0 .. maxSpeed`
8. `heading += steer * TURN_RATE * min(1, speed / 60) * dt`
9. `x += cos(heading) * speed * dt`, `y += sin(heading) * speed * dt`
10. walls: clamp `x` to `0..width`, `y` to `0..height`, and set `speed = 0` on contact
11. boost bookkeeping: if `boosting`, `boostActive = max(0, boostActive - dt)`;
    otherwise `boostCharge = min(1, boostCharge + dt / BOOST_RECHARGE)`
12. checkpoints and laps (§2.3)

**Opponents** are stepped in the same call with the same rules 2 and 4–10 (they never
boost: for them `boosting` is always false), then their own checkpoints and laps as in
§2.3 (they keep `lap` and `nextCheckpoint` but need no lap times). Each opponent's
`input` comes from **your controller**, which must be a pure function of the current state,
the opponent's index and the seed. Requirements on the controller:

- **Seeded.** Two configs that differ only in `seed` must produce different opponent states
  after 600 steps. Derive per-opponent parameters (skill, aggression, whatever you like) from
  the seed with a PRNG you implement — never `Math.random`.
- **Competent.** On the standard track (§5), with the player standing still, at least one
  opponent must complete a lap within 60 simulated seconds.

Finally, recompute `ranking` (§2.4).

### 2.3 Checkpoints, laps, finish — ordered, and this is the anti-cheat

`nextCheckpoint` starts at `1`. When a car's centre is within `r` of
`checkpoints[nextCheckpoint]`, advance it (wrapping to `0`).

**Checkpoints only count in order.** Touching checkpoint 3 while 2 is still pending does
nothing at all. A lap increments only when `nextCheckpoint` wraps from the last back to
`0`, which is what makes cutting the course impossible rather than merely discouraged. So
the lap is credited on reaching the **last** checkpoint; the start/finish circle is then
simply the next target (exactly as in racing-v1).

For the player, on each completed lap: push `t - lapStart` onto `lapTimes`, set `bestLap`
to the minimum of `lapTimes`, set `lapStart = t`; and if `lap >= laps`, set
`finished = true`.

### 2.4 `ranking`

Best first. Order by: completed laps (more is better), then checkpoints passed in the
current lap, i.e. `(nextCheckpoint - 1 + n) % n` where `n` is the number of checkpoints
(more is better), then distance to the next checkpoint (less is better); on a full tie the
player ranks ahead, then lower opponent index. `ranking` is always a permutation of
`["player", "opp0", …, "opp<opponents-1>"]`.

### 2.5 `hash(state)`

A stable string digest of the entire state. Requirements:

- identical states → identical string
- any change to any numeric field, in any car → different string
- **independent of key insertion order** (sort keys recursively; do not rely on
  `JSON.stringify` ordering)
- floats rounded to 6 decimal places before hashing, so that arithmetic that is
  mathematically equal but differs in the last bit still agrees

## 3. The camera (graded)

The file MUST also define a global `View` with one pure function — no DOM, no clock, no
`Math.random`:

```js
globalThis.View = {
  project(point, camera),   // -> { sx, sy, depth }
};
```

- `point` is `{ x, y, z }` in world units: `x`, `y` are the track plane exactly as in
  `SimCore`; `z` is height above it (the track is `z = 0`).
- `camera` is `{ x, y, z, yaw, pitch, fov, width, height }`: position; `yaw` in radians in
  the track plane, with the same convention as a car's `heading` (looking along
  `(cos yaw, sin yaw)`); `pitch` in radians, positive tilts the view up, negative down;
  `fov` the **vertical** field of view in radians; `width`, `height` the viewport in pixels.
- It is a **perspective (pinhole) projection**. `sx`, `sy` are pixel coordinates with
  `(0, 0)` at the top-left of the viewport; `depth` is the distance along the camera's
  viewing axis.
- The camera's right-hand side is the direction `yaw + π/2` — the side a car turns toward
  with `steer = +1` — and it lands on the **right** of the viewport (`sx > width / 2`).

Exactly what is checked:

1. a point on the viewing axis, ahead of the camera, projects to the viewport centre
   `(width / 2, height / 2)` with `depth > 0`
2. of two points ahead of the camera at the same lateral offset, the farther one has the
   greater `depth` and lands **closer** to the horizontal centre (foreshortening — this is
   what rules out an orthographic or top-down "3D")
3. a point behind the camera returns `depth <= 0`
4. `yaw` follows the heading convention above, negative `pitch` looks down (a point
   ahead on the track plane rises on screen when the camera tilts down), and the
   right-hand side is the right of the viewport

Your renderer should use this function (it is how your 3D view becomes checkable), but
what it draws with it — road, kerbs, scenery, sky, the other cars, a minimap — is yours.

## 4. Determinism (graded, and the hardest requirement here)

Given the same `config` and the same sequence of inputs, `SimCore` must produce a
**bit-identical** state after any number of steps. That means:

- no `Math.random()`, no `Date.now()`, no `performance.now()` inside `SimCore` or `View`
- no iteration over anything whose order is not defined
- no dependence on wall-clock time or frame rate — `dt` is a parameter, never measured

The opponents are part of the state and bound by all of the above.

## 5. The game (judged by people, not by machine)

Beyond the above you decide everything: how the 3D world looks, how it feels to drive,
what the road and its surroundings look like, how the boost and off-road surfaces are
communicated, a HUD with lap times and position, a minimap, sound, a menu, weather.

Controls must be discoverable without instructions. Arrow keys or WASD, plus one key for
boost, are expected.

**The standard track** the automated checks use (your game may ship any tracks it likes):

```
width 1000, height 800, roadWidth 120
centerline  (150,150) (850,150) (850,650) (150,650)
checkpoints (150,400,r40) (500,150,r40) (850,400,r40) (500,650,r40)
laps 3, opponents 2
```

## 6. What is checked automatically

The automated floor is pass/fail and only asks whether you built a working thing:

1. `game.html` exists, is under 256 KB, and references no external URL
2. `SimCore` and `View` load in a bare JS context — no DOM, no browser globals
3. `create()` is pure: same config twice → identical `hash`
4. `step()` matches the specified physics on hand-computed cases: on the road, off the
   road, boosting, and the boost's activation, duration and recharge rules
5. `step()` does not mutate its argument; the speed clamp is actually reached; walls stop
   the car
6. checkpoints only advance in order; a lap increments exactly once per full ordered
   circuit; `lapTimes`, `bestLap` and `finished` follow §2.3; a finished state is frozen
7. opponents: the configured count; seeded (different seed → different state); competent
   (a lap within 60 s on the standard track); within the world; never above `MAX_SPEED`
8. `ranking` is always a permutation and a car with more laps ranks first
9. determinism: the same 600-step input tape produces the same final hash, twice
10. the state stays JSON-serialisable throughout
11. `View.project` satisfies §3

Failing any of these means the submission does not enter human judging at all. Passing
them says nothing about whether the game is any good — that is the other half, and it is
judged blind by people playing it.
