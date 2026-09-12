# racing-v2 — City Run: a timed 3D points race with traffic

Build a playable 3D driving game as **one self-contained HTML file**.

This spec is the whole brief. It is deliberately precise about the world, the rules and
the projection, and deliberately quiet about how any of it looks: everything in §2–§4 is
graded by machine, everything in §5 is judged by people playing your game. Where this
document does not constrain you, you are free — and that freedom is the part being
evaluated.

---

## 1. Deliverable

A single file, `game.html`. It must:

- open and play in a modern browser by double-clicking it — **no build step, no server,
  no network access at runtime, no external assets, no CDN, no external libraries**
- be under 256 KB
- render a **3D perspective view** (a chase camera behind the player's car, or a cockpit
  view) to a `<canvas>` — WebGL or your own software projection onto a 2D canvas are
  both fine; a top-down 2D view does not satisfy this spec
- be playable with the keyboard, with a start screen (§1.2) reachable by keyboard alone

Everything — markup, styles, code — lives in that one file. Embed any image or audio you
want as a `data:` URI, or generate it procedurally.

### 1.1 The game in one paragraph

You drive a car through a small city for a fixed time — 30 seconds, 60 seconds or
2 minutes, chosen before the run. Scattered across the city is a set of **pickups**. Each
one you reach is worth its point value; when the last one is taken they all reappear and
the round counter goes up. Computer-driven cars are out there collecting the same pickups,
and **every collision with another car costs you points**. Roads are fast; anything off the
asphalt is slow. A boost gives a short burst of speed and then recharges. When the clock
runs out, the run is over and the scores are ranked.

### 1.2 The start screen — choosing the road, the car and the time

Before the run starts the player chooses three things, by keyboard:

| choice | options | source |
|---|---|---|
| **the road** (which city map to drive) | at least two maps of your own design | `SimCore.presets().tracks` |
| **the car** | exactly the three cars of §2.5 | `SimCore.presets().cars` |
| **the time limit** | `30`, `60` or `120` seconds | `SimCore.presets().times` |

How the screen looks is yours. What is graded is that these options exist, are the ones
the pure `presets()` function reports (§2.6), and that a run can be started with any
combination of them. During the run the HUD is expected to show at least the time left,
the score and the position; when time is up, an end screen with the final score and
position is expected. Both are judged by people, not by the machine.

## 2. The simulation core (graded)

The file MUST define a global `SimCore` with exactly this shape. An automated check drives
it headlessly, with no DOM, so **`SimCore` must not touch `window`, `document`, `canvas`,
timers, or `Math.random` at module scope or inside any function below.**

```js
globalThis.SimCore = {
  presets(),                       // -> { tracks, cars, times }  (pure, constant)
  create(config),                  // -> state   (pure; same config => same state)
  step(state, input, dt),          // -> state   (pure; returns NEW state, never mutates)
  hash(state),                     // -> string  (stable digest of the full state)
};
```

### 2.1 The world: a city map, its roads and its surfaces

A map is `track = { width, height, roadWidth, roads, spawn, pickups }`.

- `width`, `height` — the world rectangle, in world units. Nothing may leave it: a car that
  reaches an edge is clamped to it and stopped (§2.3 rule 10).
- `roads` — an array of at least one **polyline**, each an array of at least two `{ x, y }`
  points. A polyline is a street: consecutive points are joined by straight segments. A
  street that repeats its first point at the end is a loop. Streets may cross; the
  crossing is an intersection, and nothing special happens there.
- `roadWidth` — every street is this wide. **The road surface is every point whose
  distance to the nearest segment of any street is at most `roadWidth / 2`** (the closest
  point on a segment, including its endpoints — so streets have rounded ends).
  **Everything else is off-road**: pavement, grass, plaza, whatever you draw it as. There
  are only these two surfaces, and a position is on exactly one of them.
- `spawn` — `{ x, y, heading }`: where the player starts and the direction they face
  (radians; `0` is +x, `π/2` is +y, the same convention as a car's `heading`). It is on
  the road.
- `pickups` — an array of at least six `{ x, y, r, value }` circles: the points the cars
  are trying to reach. All of them are on the road. `value` is the points awarded, ≥ 1.

**Where the cars are on the surface.** A car is a point `(x, y)` on the plane — its
centre — with a `heading` and a `speed`, plus a collision circle of radius `CAR_RADIUS`
around the centre (§2.4). Whether the car is on the road is decided by its centre alone,
using the rule above, at the position it had **at the start of the step**. The player
spawns at `spawn`; opponent `i` spawns `40 * (i + 1)` units ahead of the spawn along the
spawn heading (so `x + cos(heading) * 40 * (i + 1)`, likewise for `y`), facing the same
way. That places every car on the road, a car-length apart, with nobody overlapping.

### 2.2 `create(config)`

`config` is `{ seed, time, car, opponents, track }`.

- `seed` — a 32-bit unsigned integer. **All randomness must derive from it.** A call to
  `Math.random()` anywhere in `SimCore` is a defect.
- `time` — the time limit in seconds: `30`, `60` or `120`.
- `car` — the player's car, an index into the cars table of §2.5.
- `opponents` — number of computer-driven cars, ≥ 0.
- `track` — a map as in §2.1 (one of your presets, or any other valid map: the automated
  check hands in its own).

Returns a `state`. `state` must be JSON-serialisable — no functions, no `Map`, no `Set`,
no cyclic references — and must contain **at least** these fields (add whatever else you
need):

```
t              seconds elapsed, starts 0
timeLimit      copied from config.time
timeLeft       max(0, timeLimit - t)
finished       false until t >= timeLimit
score          the player's points, starts 0 (may go negative)
crashes        how many collisions the player has had, starts 0
round          how many times the whole pickup set has been cleared, starts 0
player         { x, y, heading, speed, car, boostCharge, boostActive }
opponents      array of `config.opponents` cars: { x, y, heading, speed, car, score, crashes }
pickups        one entry per config pickup, in order: { x, y, r, value, taken }
ranking        array of ids, best first: "player", "opp0", "opp1", …
```

Every car starts with `speed = 0`. The player's `car` is `config.car`; each opponent's
`car` is chosen from the seed (any of the three). `boostCharge` starts at `1`,
`boostActive` at `0`. Every pickup starts with `taken = false`.

### 2.3 `step(state, input, dt)` — movement

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
ACCEL          = 220.0   units/s^2 at throttle = 1, on the road, in the balanced car
BRAKE          = 400.0   units/s^2 at brake = 1
DRAG           = 0.9     multiplicative per second on the road: speed *= DRAG ** dt
TURN_RATE      = 2.6     radians/s at |steer| = 1 in the balanced car, scaled by min(1, speed / 60)
MAX_SPEED      = 320.0   units/s

OFFROAD_ACCEL  = 0.5     acceleration is multiplied by this off the road
OFFROAD_DRAG   = 0.35    replaces DRAG off the road

BOOST_MULT     = 1.8     acceleration is multiplied by this while the boost is active
BOOST_MAX      = 400.0   replaces MAX_SPEED while the boost is active
BOOST_DURATION = 2.0     seconds a boost lasts
BOOST_RECHARGE = 8.0     seconds from empty to full charge

CAR_RADIUS     = 10.0    collision circle around a car's centre
CRASH_PENALTY  = 25      points lost by EACH car in a collision
CRASH_SLOWDOWN = 0.6     see §2.4
```

Per step, in this order. Steps 1–11 are the player; `carSpec` is the player's row of §2.5.

1. `t += dt`
2. `onRoad` = the player's centre, at its position before this step, is on the road (§2.1)
3. boost: if `input.boost` is 1 **and** `boostActive == 0` **and** `boostCharge >= 1`,
   set `boostActive = BOOST_DURATION` and `boostCharge = 0`. Then `boosting = boostActive > 0`.
4. `accel = ACCEL * carSpec.accel * (onRoad ? 1 : OFFROAD_ACCEL) * (boosting ? BOOST_MULT : 1)`,
   `drag = onRoad ? DRAG : OFFROAD_DRAG`, `maxSpeed = boosting ? BOOST_MAX : MAX_SPEED`
5. `speed += (accel * throttle - BRAKE * brake) * dt`
6. `speed *= drag ** dt`
7. clamp `speed` to `0 .. maxSpeed`
8. `heading += steer * TURN_RATE * carSpec.turn * min(1, speed / 60) * dt`
9. `x += cos(heading) * speed * dt`, `y += sin(heading) * speed * dt`
10. walls: clamp `x` to `0..width`, `y` to `0..height`, and set `speed = 0` on contact
11. boost bookkeeping: if `boosting`, `boostActive = max(0, boostActive - dt)`;
    otherwise `boostCharge = min(1, boostCharge + dt / BOOST_RECHARGE)`
12. **opponents**, in index order: each is moved by rules 2 and 4–10 with its own `carSpec`,
    never boosting, with the `input` your controller returns for it (§2.7)
13. **collisions** (§2.4), over every pair of cars
14. **pickups** (§2.4): the player first, then the opponents in index order
15. `timeLeft = max(0, timeLimit - t)`; if `t >= timeLimit`, `finished = true`
16. recompute `ranking` (§2.8)

### 2.4 Collisions and pickups

**Collisions.** After every car has moved (rule 12), examine every pair of cars exactly
once, in this fixed order: `(player, opp0), (player, opp1), …, (opp0, opp1), (opp0, opp2),
…` — i.e. all pairs `(i, j)` with `i < j` over the list `[player, opp0, opp1, …]`. For a
pair whose centres are **strictly closer than `2 * CAR_RADIUS`** it is a crash:

- push them apart along the line joining their centres, **each by half the overlap**, so
  that they end exactly `2 * CAR_RADIUS` apart (if the centres coincide, push along the
  first car's heading); then clamp both to the world rectangle as in rule 10
- each car's `speed` is multiplied by `1 - CRASH_SLOWDOWN * otherMass / (ownMass + otherMass)`
  where the masses come from §2.5 — the heavier car keeps more of its speed
- each car loses `CRASH_PENALTY` points and its `crashes` goes up by one — for the player
  that is `state.score` and `state.crashes`, for an opponent its own `score` and `crashes`

Because the resolution leaves the pair exactly touching, and only strictly overlapping
pairs count, a crash is counted once per contact, not once per frame; two cars that keep
pushing into each other crash again only if they close in again.

**Pickups.** After collisions, the player and then each opponent in index order collects
every pickup that is not yet `taken` and whose centre is within `r` of the car's centre:
it becomes `taken = true` and the car's score goes up by its `value`. A pickup can only be
taken once per round. When the last pickup of the set is taken, **at the end of that step**
every pickup becomes `taken = false` again and `round` goes up by one.

### 2.5 The cars

Exactly this table, in this order; `car` in a config or a state is an index into it.

| index | id | accel | turn | mass | character |
|---|---|---|---|---|---|
| 0 | `balanced` | 1.00 | 1.00 | 1.0 | the reference car — the constants above apply as written |
| 1 | `sprint` | 1.15 | 0.90 | 0.8 | quicker off the line, turns less sharply, loses more in a crash |
| 2 | `heavy` | 0.90 | 0.85 | 1.3 | slower and wider-turning, shrugs off crashes |

`accel` multiplies `ACCEL` (rule 4), `turn` multiplies `TURN_RATE` (rule 8), `mass` sets
the crash slowdown (§2.4). `MAX_SPEED`, `BRAKE`, the drags and the boost are the same for
every car.

### 2.6 `presets()`

Returns `{ tracks, cars, times }`, the same value every time, computed from nothing:

- `tracks` — at least **two** maps of your own design, each `{ name, track }` with `track`
  valid under §2.1 (≥ 1 street, ≥ 6 pickups, all pickups and the spawn on the road).
  These are the roads the start screen offers.
- `cars` — the table of §2.5, as `[{ id, accel, turn, mass }, …]`, in order.
- `times` — `[30, 60, 120]`.

### 2.7 The opponents' controller

Each opponent's `input` for rule 12 comes from **your controller**: a pure function of the
current state (as it was at the start of the step), the opponent's index and the seed.
Requirements:

- **Seeded.** Two configs that differ only in `seed` must produce different opponent states
  after 600 steps. Derive per-opponent parameters (which car, skill, whatever you like)
  from the seed with a PRNG you implement — never `Math.random`.
- **Competent.** On the standard map (§4), with the player standing still, the opponents
  between them must take at least one pickup within 30 simulated seconds.
- Opponents never boost.

### 2.8 `ranking`

Best first. Order by `score` (more is better); on a tie the player ranks ahead, then the
lower opponent index. `ranking` is always a permutation of
`["player", "opp0", …, "opp<opponents-1>"]`.

### 2.9 `hash(state)`

A stable string digest of the entire state. Requirements:

- identical states → identical string
- any change to any numeric field, in any car or pickup → different string
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

- `point` is `{ x, y, z }` in world units: `x`, `y` are the city plane exactly as in
  `SimCore`; `z` is height above it (the ground is `z = 0`).
- `camera` is `{ x, y, z, yaw, pitch, fov, width, height }`: position; `yaw` in radians in
  the plane, with the same convention as a car's `heading` (looking along
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
4. `yaw` follows the heading convention above, negative `pitch` looks down (a point ahead
   on the ground rises on screen when the camera tilts down), `+z` is up on screen, and the
   right-hand side is the right of the viewport

Your renderer should use this function (it is how your 3D view becomes checkable), but
what it draws with it — streets, kerbs, buildings, pickups, the other cars, a minimap, the
sky — is yours.

## 4. Determinism (graded, and the hardest requirement here)

Given the same `config` and the same sequence of inputs, `SimCore` must produce a
**bit-identical** state after any number of steps. That means:

- no `Math.random()`, no `Date.now()`, no `performance.now()` inside `SimCore` or `View`
- no iteration over anything whose order is not defined
- no dependence on wall-clock time or frame rate — `dt` is a parameter, never measured

The opponents and the pickups are part of the state and bound by all of the above.

**The standard map** the automated checks use (your game ships its own presets; this one
need not be among them):

```
width 1000, height 800, roadWidth 120
roads   ring    (150,150) (850,150) (850,650) (150,650) (150,150)
        avenue  (500,150) (500,650)
        avenue  (150,400) (850,400)
spawn   (150,400) heading 0            — facing east along the middle avenue
pickups r 30, value 10, at
        (500,400) (500,150) (850,400) (500,650) (150,150) (850,150) (850,650) (150,650)
opponents 2, car 0 (balanced), time 60
```

## 5. The game (judged by people, not by machine)

Beyond the above you decide everything: how the city looks in 3D, what tells asphalt from
grass, how a pickup announces itself, how a crash feels and sounds, how the boost is shown,
the start screen, the HUD, the end screen, a minimap, weather, music.

Controls must be discoverable without instructions. Arrow keys or WASD, plus one key for
boost, are expected; the start screen should be navigable with the same keys plus Enter.

What people will be asked when they play it: *Was it clear where the roads are? Could I
tell when I was off them? Did I know where the pickups were and how long I had left? Did
a crash feel like a crash? Would I play another 60 seconds?*

## 6. What is checked automatically

The automated floor is pass/fail and only asks whether you built a working thing:

1. `game.html` exists, is under 256 KB, and references no external URL
2. `SimCore` and `View` load in a bare JS context — no DOM, no browser globals
3. `presets()` is pure and reports ≥ 2 valid maps, the exact cars table and `[30, 60, 120]`;
   a run can be created and stepped on every preset map with every car and time
4. `create()` is pure: same config twice → identical `hash`; the initial state has the
   fields of §2.2, with every car spawned as §2.1 says
5. `step()` matches the specified physics on hand-computed cases: on the road, off the road,
   at the road's edge, in each car, boosting, and the boost's activation, duration and
   recharge rules; the speed clamp is actually reached; walls stop the car
6. `step()` does not mutate its argument
7. collisions: a strictly overlapping pair is pushed to exactly `2 * CAR_RADIUS` apart, both
   speeds are scaled by the mass rule, both lose `CRASH_PENALTY` and count a crash; the
   pair does not crash again on the next step
8. pickups: reaching one scores its `value` once; when the set is cleared it resets and
   `round` increments
9. the timer: `finished` flips exactly when `t` reaches the limit, `timeLeft` reaches 0,
   and a finished state is frozen
10. opponents: the configured count; seeded; competent (a pickup within 30 s on the
    standard map); within the world; never above `MAX_SPEED`
11. `ranking` is always a permutation, ordered by score with the tie rule
12. determinism: the same 600-step input tape produces the same final hash, twice
13. the state stays JSON-serialisable throughout
14. `View.project` satisfies §3

Failing any of these means the submission does not enter human judging at all. Passing
them says nothing about whether the game is any good — that is the other half, and it is
judged blind by people playing it.
