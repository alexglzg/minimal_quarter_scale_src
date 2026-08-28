# Integration guide — feeding inputs to the amortized model

This document is for someone wiring `anmpc_alpha.py` into their own stack (e.g. a ROS
node). It describes **exactly** what the controller expects, what it returns, and the
conventions you have to honour so the model stays inside the regime it was trained for.

`README.md` covers what the bundle is; this file covers how to call it.

---

## 1. The call in one block

```python
import numpy as np, jax.numpy as jnp, equinox as eqx
from anmpc_alpha import load_alpha_model

model, dtype = load_alpha_model("model.eqx", meta_path="model_meta.yaml")

@eqx.filter_jit                    # compile once, reuse forever
def controller(m, x, path, obs_centers, obs_radii):
    return m(x, path, obs_centers, obs_radii)

u_safe, u_nom, gammas = controller(model, x, path, obs_centers, obs_radii)
u = np.asarray(u_safe, float)      # (4,) thruster forces, ready to publish
```

The model is **stateless** — one call maps one observation to one command. All memory
(the arc-length `s`) lives on your side; see §5.

---

## 2. Inputs — shapes, dtype, units

Everything is float64 (`dtype` returned by the loader is `jnp.float64`; the module sets
`jax_enable_x64` at import). Pass `jnp.asarray(a, dtype)`; plain numpy arrays also work
but get converted on every call.

| argument | shape | dtype | contents |
|---|---|---|---|
| `x` | `(7,)` | f64 | `[X, Y, psi, su, sv, sr, s]` |
| `path` | `(3,)` | f64 | `[kx, y0, u_ref]` |
| `obs_centers` | `(n_obs, 2)` = `(6, 2)` | f64 | obstacle centres in world frame `[m]` |
| `obs_radii` | `(n_obs,)` = `(6,)` | f64 | obstacle radii `[m]` |

**Batching is not built in.** The signature is per-sample; use `jax.vmap` if you ever
need many at once. A ROS node needs one call per control tick.

### 2.1 State vector `x` (7,)

| idx | symbol | frame | unit | notes |
|---|---|---|---|---|
| 0 | `X` | world | m | position |
| 1 | `Y` | world | m | position |
| 2 | `psi` | world | rad | heading, **wrapped to `[-pi, pi)`** — use `wrap_angle` |
| 3 | `su` | body | m/s | surge (forward) velocity |
| 4 | `sv` | body | m/s | sway (lateral, positive to port/left) velocity |
| 5 | `sr` | body | rad/s | yaw rate |
| 6 | `s` | — | m-ish | path arc-length parameter, see §5 |

Body axes follow the standard Fossen convention: `fwd = [cos psi, sin psi]`,
`left = [-sin psi, cos psi]`. If your odometry publishes world-frame velocities
`(Vx, Vy)`, convert before calling:

```python
su =  Vx*np.cos(psi) + Vy*np.sin(psi)
sv = -Vx*np.sin(psi) + Vy*np.cos(psi)
```

`psi` is a *nautical/ENU heading measured CCW from +X*, not a compass bearing. If your
source is compass degrees, convert (`psi = np.deg2rad(90 - bearing_deg)`) and wrap.

### 2.2 Path `path` (3,)

The reference is a fixed sinusoid parameterized by `s`:

```
x_d(s) = kx * s
y_d(s) = sin(kx * s) + y0
```

| idx | symbol | unit | meaning |
|---|---|---|---|
| 0 | `kx` | 1/m | frequency/scale of the sinusoid |
| 1 | `y0` | m | vertical offset of the path |
| 2 | `u_ref` | m/s | desired surge speed |

This shape is **baked into the model**: both the learned features and the Gauss-Newton
analytic baseline evaluate `x_d, y_d` and the tangent angle analytically. You cannot
feed an arbitrary waypoint list or a spline through this argument. To follow a
different geometry you would have to retrain (or locally fit a sinusoid, which is not
something this bundle supports).

### 2.3 Obstacles

Circular only, in the **world** frame, same units as `X, Y`. The model converts them to
body coordinates internally, so do **not** pre-transform them.

> **Where that conversion happens, and why it only affects half the pipeline**
>
> The transform lives in `_features()` in `anmpc_alpha.py`: each centre is made relative
> to the vessel (`c - p`) and projected onto the body axes `fwd = [cos psi, sin psi]`,
> `left = [-sin psi, cos psi]`. Those `2*n_obs` numbers plus the raw radii are the
> `3*n_obs` obstacle features appended to the 11 base features.
>
> This is **only** the network input. The CBF rows are built in the world frame from the
> raw centres you pass — `_cbf_obs_jax` / `_cbf_single` evaluate
> `h = ||x[:2] - c||^2 - r_safe^2` on world positions directly. So the body transform
> shapes the predicted QP *objective*, never the safety *constraint*. That is exactly why
> the interface wants world coordinates: the code converts where a body frame helps and
> needs the originals where it doesn't.
>
> Body frame is used for the features because the correct avoidance response depends on
> where an obstacle sits *relative to the heading* (2 m dead ahead vs. 2 m abeam), not on
> its absolute position. Relative offsets make the same geometry look identical anywhere
> along the path and at any heading, and they stay in a few-metre range, whereas world `X`
> grows without bound as the vessel advances. The padding table below is that scaling
> argument being measured. Absolute `X, Y` and `cos psi, sin psi` are still kept among the
> base features, since the reference path is itself defined in world coordinates.

`n_obs = 6` is **structural**: it fixes the feature dimension (`11 + 3*n_obs = 29`), the
number of CBF rows in the QP, and the width of the two gamma heads. Passing a different
count raises a shape error at trace time — it is not a soft parameter.

**If you have fewer than 6 obstacles**, pad by **duplicating a real obstacle** — repeat
the farthest one into the unused slots. The duplicate CBF rows are redundant constraints,
which the QP handles without trouble, and the feature vector stays in-distribution.

Do **not** park unused slots far away (e.g. `(-50, -50)`), even though their CBF rows stay
inactive. Obstacle positions are also *network inputs*: the feature vector carries each
obstacle's body-frame offset, and a 50 m offset is far outside anything seen in training.
Measured over a 400-step rollout on the same in-distribution scenario:

| padding for 3 unused slots | cross-track RMS | saturated ticks | min clearance |
|---|---|---|---|
| 6 real obstacles (no padding at all) | 0.63 m | 0 % | 0.043 m |
| **duplicate the farthest real obstacle** | **0.64 m** | **0 %** | 0.046 m |
| park spares 12 m behind, off to the side | 0.83 m | 0 % | 0.164 m |
| park spares at `(-50, -50)` | 1.69 m | 35 % | 0.508 m |

Duplication is indistinguishable from having six genuine obstacles; distant parking costs
~2.6× the cross-track error and saturates the thrusters a third of the time. Both remain
collision-free — safety comes from the CBF rows — but path-following quality degrades
badly. `integration_walkthrough.ipynb` reproduces this A/B in two rollouts if you want to
re-check it.

The general rule: anything you feed the model is a network input, not just a constraint.
"Harmless because the constraint is inactive" is not the same as "harmless". If you must
park rather than duplicate, keep spares within a plausible distance (~10–15 m behind, off
to the side) rather than hundreds of metres away — and never ahead on the path.

**If you have more than 6**, select the 6 most relevant each tick — nearest by
along-path distance ahead of the vessel is the natural rule. Beware that the selection
can flip between ticks and cause a command jump; hysteresis (keep a selected obstacle
until it is clearly behind) is worth adding.

**Ordering matters.** The features concatenate per-obstacle body-frame offsets in the
order you pass them, so the network is *not* permutation-invariant. Training data always
listed obstacles in increasing along-path order. Sort your obstacles the same way
(ascending projection onto the path / ascending `X`) and keep that order stable across
ticks.

### 2.4 Values the model was trained on

Stay near these; the model degrades outside them and the safety guarantee comes from the
QP's CBF rows, not from the network being in-distribution.

| quantity | training range |
|---|---|
| `kx` | 0.10 – 0.30 |
| `y0` | 2.5 – 3.5 |
| `u_ref` | 0.40 – 0.70 m/s |
| obstacle radius | 0.4 – 0.8 m |
| obstacle along-path spacing | 4 – 8 m (first one 2.5 m from start) |
| obstacle lateral offset from path | ±0.2 m |
| initial `su` | 0 – 0.5 m/s, `|sv|, |sr|` ≤ 0.05 |
| heading error at start | ±0.6 rad |
| workspace | X, Y roughly in `[-4, 65]` |

Fixed platform constants (in `model_meta.yaml` / `anmpc_alpha.py`): `dt = 0.1 s`,
`f_max = 6.0 N`, ego bounding box `0.90 × 0.45 m` → `r_ego = hypot(L,W)/2 ≈ 0.503 m`.
The CBF uses `r_safe = obs_radius + r_ego`, so obstacle radii you pass should be the
*true obstacle* radii — do not pre-inflate them by the vessel size.

### 2.5 `r_ego` — the one dimension you *can* change freely

Unlike `n_obs`, `dt` or the cost weights, `r_ego` is not baked into anything learned. It
appears at exactly one place in `anmpc_alpha.py`:

```python
r_safe = obs_radii + r_ego        # inside the CBF row builder
```

It is absent from the feature vector and from the Gauss-Newton analytic baseline, so **no
network weight depends on it**. Pass it at load time:

```python
model, dtype = load_alpha_model("model.eqx", meta_path="model_meta.yaml", r_ego=0.9)
```

Measured over the standard 500-step scenario, sweeping only `r_ego` (hull clearance is
always computed against the *true* 0.503 m hull, so the rows are comparable):

| `r_ego` | hull clearance | margin vs its own bound | cross-track RMS | progress |
|---|---|---|---|---|
| 0.30 | **−0.117 m** (real collision) | 0.086 m | 0.60 m | 17.3 m |
| 0.503 (as trained) | 0.043 m | 0.043 m | 0.67 m | 17.3 m |
| 0.70 | 0.223 m | 0.026 m | 0.76 m | 17.3 m |
| 1.00 | 0.513 m | 0.016 m | 0.93 m | 17.2 m |
| 1.50 | 1.005 m | 0.008 m | 1.24 m | 16.3 m |
| 2.00 | 1.497 m | 0.000 m | 1.52 m | 19.1 m |

Read three things off that table:

1. **Raising it buys standoff nearly one-for-one.** `r_ego = 1.0` yields 0.51 m of true
   clearance — almost exactly the 0.497 m you added. This is the clean way to demand a
   safety buffer.
2. **The QP always respects whatever bound you declare** (middle column never goes
   negative), and no thruster saturated at any setting. Note the margin shrinking toward
   zero: the larger `r_ego` gets, the more the vessel rides exactly on its own boundary,
   because the constraint is active almost continuously.
3. **You pay in tracking**, not in safety: cross-track RMS grows from 0.67 m to 1.52 m
   across the sweep. The network is still proposing an objective calibrated for
   `r_ego = 0.503`, so the QP has to overrule it more and more. Progress holds up
   throughout — no stalls even at 4× the trained radius.

**Lowering `r_ego` below the true hull radius is the dangerous direction.** At 0.30 the
controller is perfectly "safe" with respect to the radius it was given, and the real hull
still hits the obstacle (−0.117 m). The CBF guarantees what you declare, not what is true.

Why change `r_ego` rather than inflate `obs_radii`? Only the *sum* `obs_radius + r_ego`
enters the constraint, so either route moves the barrier identically — but `obs_radii` is
also a **network input** (the last `n_obs` feature entries), while `r_ego` is not.
Inflating radii to 1.5 m pushes them far outside the trained 0.4–0.8 m band and degrades
the prediction; raising `r_ego` leaves the features untouched. For pure safety margin,
`r_ego` is the cleaner knob.

**What you cannot fix this way:** if your vessel is genuinely a different size, `r_ego`
corrects the *geometry* only. The Fossen dynamics — masses `M11/M22/M33`, drag
`D11/D22/D33`, thruster moment arms, `f_max` — are hard-coded module constants and were
baked into training. A materially different hull needs retraining, not a bigger `r_ego`.

---

## 3. Outputs

```python
u_safe, u_nom, gammas = model(x, path, obs_centers, obs_radii)
```

| return | shape | meaning |
|---|---|---|
| `u_safe` | `(4,)` | thruster forces `[f1, f2, f3, f4]` in N — **this is the command** |
| `u_nom` | `(4,)` | always zeros for this model; a visualization hook, ignore it |
| `gammas` | `(6, 2)` | per-obstacle learned HOCBF class-K gains `[gamma1, gamma2]` — diagnostic only |

`u_safe` already satisfies `|f_i| <= f_max` (box rows inside the QP), so no clipping is
needed; clipping anyway as a belt-and-braces guard is harmless.

Thruster layout (`f1, f2` surge pair, `f3, f4` sway pair, moment arms `a = 0.45`,
`b = 0.90`):

```
tau_surge = f1 + f2
tau_sway  = f3 + f4
tau_yaw   = (a/2)*(f1 - f2) + (b/2)*(f3 - f4)
```

If your vehicle has a different allocation, map through these generalized forces rather
than sending `f1..f4` directly.

`gammas` is worth logging: gains collapsing toward the 0.01 floor, or `u_safe`
saturating on many consecutive ticks, are the practical signals that the scenario is
harder than what the model saw.

---

## 4. Timing

Measured on this CPU-only machine (float64, `eqx.filter_jit`, `n_obs = 6`):

- first call: **~4.3 s** (JAX tracing + XLA compilation)
- steady state: **mean 1.5 ms**, p95 2.1 ms, max 2.2 ms per step

So a 10 Hz loop (the `dt = 0.1 s` the model was trained at) has enormous margin. Two
practical consequences for a ROS node:

- **Warm up in the constructor**, before you start publishing — call the controller once
  on a dummy observation and `jax.block_until_ready(...)` the result. Otherwise your
  first control tick blows its deadline by 4 seconds.
- **Keep shapes and dtypes constant.** Any change in `n_obs`, dtype, or the
  numpy-vs-jax-ness of an argument retriggers compilation and costs seconds mid-run.
  Allocate the input buffers once and refill them.

Run at `dt = 0.1 s`. The model has no explicit `dt` input, but `dt` enters the
linearization inside the analytic baseline and was fixed at 0.1 during training.
Running the loop at a different rate is a silent model mismatch, not an error.

Threading: JAX is not reentrant in a way worth relying on — call the controller from a
single thread/timer callback.

---

## 5. The arc-length `s` — the one piece of state you own

`x[6]` is the path parameter, not a sensor reading. It must be re-anchored to the
vessel's actual position every tick, otherwise the reference drifts away from the boat
and the cross-track term becomes meaningless:

```python
from anmpc_alpha import anchor_arc_length
s = anchor_arc_length(path, x, s_prev)   # nearest projection, searched in
                                         # [s_prev - 1.0, s_prev + 10.0]
x[6] = s
```

Keep `s_prev` in your node and seed it with `anchor_arc_length(path, x0, 0.0)` at start.
The search window is deliberately local (`look_back=1.0`, `look_ahead=10.0`) so the
projection cannot jump backwards onto a previous lobe of the sinusoid — that locality is
what makes it well-defined, so don't widen it casually.

Per-tick order of operations in a ROS callback:

```python
psi = wrap_angle(psi)                       # 1. wrap heading from odom
s   = anchor_arc_length(path, x, s)         # 2. re-anchor arc length
x[6] = s
u = np.asarray(controller(model, jnp.asarray(x, dtype),
                          path_j, oc_j, or_j)[0], float)   # 3. call model
publish(u)                                  # 4. publish thruster forces
```

Note the pattern in `run_vessel_demo.py`: it anchors *before* the model call and again
*after* the plant step. In a real loop, once per tick before the call is what matters.

---

## 6. Sanity checklist before going live

- [ ] `obs_centers.shape == (6, 2)` and `obs_radii.shape == (6,)`, every tick.
- [ ] Unused obstacle slots filled by **duplicating** a real obstacle, not parked at a
      far-away sentinel position (§2.3).
- [ ] Obstacles in **world** frame, sorted along-path, radii **not** inflated by `r_ego`.
- [ ] `psi` wrapped to `[-pi, pi)`; `su, sv, sr` in the **body** frame.
- [ ] `x[6]` re-anchored via `anchor_arc_length` every tick.
- [ ] Controller warmed up before the first published command.
- [ ] Loop running at 10 Hz (`dt = 0.1`).
- [ ] `path` values inside the trained ranges (§2.4).
- [ ] Logging `gammas` and `u_safe` saturation for post-run diagnosis.

## 7. What the model does *not* do

- No sensing, tracking, or obstacle association — it takes the obstacle list as truth.
- No recovery behaviour: if the vessel starts inside an obstacle, the CBF constraint is
  already violated and the QP output is not meaningful. Guard for that upstream.
- No terminal/goal handling: it follows the sinusoid forever; deciding when the mission
  is over is your logic.
- No feasibility fallback. `qpax` runs a fixed 50-iteration interior-point solve and
  always returns *something*; it does not raise on an infeasible QP. If your scenario
  can produce conflicting CBF rows (obstacles boxing the vessel in), add a plausibility
  check on `u_safe` (finite, within bounds, not wildly oscillating) and a safe fallback.

---

## 8. Seeing what the model replaced

If you want to understand *why* the interface looks the way it does, read
`mpc_oracle.py`: it is the nonlinear MPC this model was trained to imitate, reproduced
self-contained (needs `pip install casadi`). The cost function there is what the network
learned to encode into the QP objective `(H, c)`, and its HOCBF constraint is the one
piece the learned controller still solves exactly rather than approximating — compare it
with `anmpc_alpha._cbf_single` to see the correspondence.

`compare_to_oracle.py` then runs both on the same scenarios: ~47× faster per step, with
open-loop imitation RMSE around 0.15 N (~7% of the mean command) and comparable
closed-loop safety and tracking.
