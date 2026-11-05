# Implementation guide — the quadrotor CoSTA pipeline, step by step

You implement `src/quadrotor/` yourself, in the order below. Each step says **what** to build,
**why** it exists (the exam answer), **how** (formulas, shapes, pitfalls), and **which tests prove it**.
The tests in `tests/` are the specification: run them after each step.

```bash
uv run pytest -q tests/test_labels.py
```

A finished reference implementation is saved in `reference_impl/` (git-ignored). Use it only
when you are stuck after trying; reading it first defeats the purpose.

---

## 0. The whole pipeline on one page

```
NeuroBEM CSV ──dataset──▶ pieces (gap-free) ──▶ x (N,13), u (N,4), acc, alpha
                                                │
            labels: measured wrench  F = m·acc,  T = J·α + w×Jw
                    residual = measured − motor_wrench(u)        (CoSTA, linear drag)
                    full     = measured                          (DDM)
                    features = [v_body, w, Ω]                    (no position, no time)
                                                │
models: every model is a WrenchModel  wrench(x, u) -> (F_body, T_body)
        PBM = motor_wrench · LinearDrag = motor + D v_b · CoSTA = motor + net · DDM = net
                                                │
make_dynamics(model) = rigid_body(x, model.wrench(x, u))   ← ONE physics, shared
                                                │
integrators: rk4_step / rollout (batched over windows)      ← ONE integrator, shared
                                                │
evaluate: correction / one-step / horizon errors from measured start states
train:    kf fit (train only), linear drag (lstsq), one network loop (CoSTA + DDM),
          checkpoint chosen on VALIDATION ROLLOUTS
scripts/run.py: orchestration + results/<run_id>/
```

The single design idea: **models differ only in the wrench they supply** (the course vessel notebook's
table "models differ only in the acceleration"). Everything else is written once, so the comparison is
fair by construction. That is your answer to "how did you make PBM, DDM and CoSTA comparable?".

**Batching.** Every function takes a leading batch dimension: `x` may be `(13,)` or `(B, 13)`.
Use `x[..., 6:10]`, `axis=-1`, `np.einsum("...ij,...j->...i", R, F)`. RK4 then advances all rollout
windows of a piece at once and the Python loop runs over time steps only (the course batches over
maneuvers the same way). This is a ~30× speed-up and changes no result.

---

## 1. `physics.py` — split the rigid body from the motors, and batch it

You already have `quat_to_rotmat`, `quat_mul`, `motor_wrench`, `f_pbm`, `load_params`.

**What to add / change**
1. `rigid_body(x, force_b, torque_b, p) -> xdot`: the body of your `f_pbm`, but taking the wrench as
   input. Then `f_pbm` = `motor_wrench` (+ corrections) → `rigid_body`. Keep `f_pbm`'s signature.
2. Batch `quat_to_rotmat` (`(..., 4) -> (..., 3, 3)`): take `q[..., 0]` etc. and build the matrix with
   two nested `np.stack(..., axis=-1)` / `axis=-2`.
3. Batch `quat_mul`: scalar part `s1*s2 - np.sum(v1*v2, axis=-1, keepdims=True)` with `s = q[..., :1]`.
4. Batch `rigid_body`:
   - `R @ F` → `np.einsum("...ij,...j->...i", R, F)`;
   - `p.inertia @ w` → `w @ p.inertia.T` (works for `(3,)` and `(B, 3)`);
   - `np.linalg.solve(J, b)` → `np.linalg.solve(J, b[..., None])[..., 0]`;
   - `np.concatenate(..., axis=-1)`, and `np.zeros_like(w[..., :1])` for the quaternion's 0.
5. `body_velocity(x) = R(q)ᵀ v`: `np.einsum("...ji,...j->...i", R, x[..., 3:6])` (note `ji`: transpose).

**Why:** `rigid_body` is the physics every model shares; `motor_wrench` is the physics only the
PBM-based models use. The DDM gets `rigid_body` but not `motor_wrench` (state this in the report:
your DDM learns the wrench, the course's DDM learns the whole acceleration).

**Tests:** your own `tests/test_physics.py` (hover → 0, zero correction bit-for-bit), plus
`test_models.py::test_body_velocity_matches_scipy`.

## 2. `integrators.py` — one line for batching

`x_next[..., 6:10] /= np.linalg.norm(x_next[..., 6:10], axis=-1, keepdims=True)`.
Without `keepdims` the division broadcasts wrongly for `(B, 13)`. `rollout` stays single-window.

**Tests:** your own `tests/test_integrators.py`.

## 3. `dataset.py` — from CSV columns to model arrays

Already there: loading, `continuous_pieces`, `window_starts`, splits, `fraction_outside_box`.

**What to add**
1. `STATE_COLS`, `MOTOR_COLS` and `states_inputs(piece) -> (x, u)`. The CSV quaternion is
   `(x, y, z, w)`; your state wants `[w, x, y, z]`. Order the columns accordingly and **renormalize**
   the quaternion (logged quaternions are not exactly unit length).
2. `measurements(piece) -> (acc, alpha)`: `acc_*` (body specific force) and `alpha_*`.
3. `Piece` (frozen dataclass: `flight, x, u, acc, alpha, dt`) and
   `role_pieces(raw_dir, flights, min_height, min_rows)`: for every segment take its `sample_period`
   (before dropping rows), cut it into `continuous_pieces`, keep pieces with ≥ `min_rows` rows.

**Why:** rollouts, filters and finite differences must stay inside one gap-free piece, and the
~164 Hz flight needs its own dt. `Piece` carries both.

**Check by hand on one piece:** `R(q) @ v_body_csv ≈ v_world` to 1e-15. If not, the quaternion order is wrong.

## 4. `labels.py` — what the networks learn

1. `measured_wrench(acc, alpha, w, p)`: `F = m·acc`, `T = α Jᵀ + w × (w Jᵀ)` (row-wise for `(N, 3)`).
   Why `+ w × Jw`: the PBM is `J ẇ = τ − w × Jw`, so the torque that actually acted is `Jα + w × Jw`.
   Why no gravity in `F`: `acc` is the *specific force* (what an accelerometer measures), already
   without gravity.
2. `residual_labels(x, u, a_meas, alpha_meas, p)` = measured − `motor_wrench(u)`. **Always the
   residual directly** — never `L̃(u_ref − ũ)` (L04: only valid for linear operators).
3. `targets(model_name, x, u, a_meas, alpha_meas, p) -> (N, 6)`: residual for `costa` / `linear_drag`,
   full measured wrench for `ddm`, `ValueError` otherwise.
4. `features(x, u) -> (..., 10)`: `[body_velocity(x), x[..., 10:13], u]`. No position, no time
   (L04 appendix: a time input was 50× worse past the training horizon).

**The key test** (`test_residual_is_zero_when_data_comes_from_the_pbm`): generate acc/alpha from
`f_pbm` itself; the residual must be zero. This catches every frame and sign error at once.

## 5. `models.py` — four models, one interface

1. `WrenchModel` (a `typing.Protocol`): `name: str` and `wrench(x, u) -> (force, torque)`.
2. `make_dynamics(model, p)` returns `lambda x, u: rigid_body(x, *model.wrench(x, u), p)`.
3. `PBM(p)`: `motor_wrench(u, p)`.
   `LinearDrag(p, D)`: motor force + `body_velocity(x) @ D.T` (this form batches; `D @ v` does not).
4. `Normalizer(x_mean, x_scale, y_mean, y_scale)` with `fit(X, Y, x_fallback, min_std)`: std per
   channel, but a channel with std < `min_std` gets its **declared physical scale** (L04: a zero-std
   channel produced inputs of −6×10¹⁰ in the course vessel example). Methods `scale_in`,
   `unscale_out`, `scale_out`. Fitted on TRAIN rows only, then frozen.
5. `MLP(layers, activation)`: Linear + SiLU/tanh, **no activation on the last layer**, Xavier-normal
   weights, zero biases (same as the course's `pyutils/nets.py`).
6. `FrozenMLP.from_torch(net)` copies weights to numpy (`.detach().double().numpy()`);
   `__call__(z)` = `z @ W.T + b` per layer. SiLU in numpy: `z * scipy.special.expit(z)`
   (`z / (1 + exp(-z))` overflows). Why numpy: rollouts call the net thousands of times; numpy on
   frozen weights is much faster and cannot change the weights.
7. `CoSTA(p, net, norm)`: `correction = norm.unscale_out(net(norm.scale_in(features(x, u))))`;
   `wrench = motor + c[..., :3], motor_torque + c[..., 3:]` (`[...,` so it batches).
   `DDM(net, norm)`: the same network output **is** the wrench.
8. `save_model(model, path)` / `load_model(path, p)`: one `.npz` with `kind`, `D` or the layer
   weights + activation + normalizer arrays.

**Tests:** `tests/test_models.py`. The important one: a CoSTA whose last layer is zero reproduces the
PBM **rollout bit for bit** (L04: "the PBM is always one switch away" — and why that does NOT mean
CoSTA is never worse than the PBM).

## 6. `evaluate.py` — the three errors of L04

Constants: `STATE_ERRORS = ("pos_m", "vel_mps", "att_rad", "rate_radps")`,
`WRENCH_COMPONENTS = ("F_x", ..., "T_z")`.

1. `attitude_error(q1, q2)`: angle of `conj(q1) ⊗ q2` as `2·atan2(‖vector‖, |scalar|)`.
   **Do not use `2·arccos(|q1·q2|)`**: near zero it is badly conditioned (gives ~1e-8 rad for identical
   quaternions). `|scalar|` makes it sign-invariant (q and −q are the same rotation).
2. `state_errors(x_pred, x_meas) -> (..., 4)`: norms of the position, velocity, rate differences + attitude angle.
3. `correction_error(pred, label)`: RMSE per wrench component.
4. `one_step_predictions(x, u, f, dt, stride)`: one `rk4_step` from all start rows **as one batch**;
   `one_step_error`: RMSE of `state_errors` vs `x[1:][::stride]`.
5. `_window_errors(x, u, f, dt, n_steps, stride) -> (n_windows, n_steps+1, 4)`: `starts = window_starts(...)`,
   `X = x[starts]`, then for each step `k`: `X = rk4_step(f, X, u[starts + k], dt)` and store
   `state_errors(X, x[starts + k + 1])`. Non-finite → `inf`; wrap in `np.errstate(all="ignore")`.
6. `horizon_rollout_error(..., horizons_s, stride)`: one call with the longest horizon, then read off
   each horizon's step — **the same window starts for every horizon**.
7. `validation_score(f, pieces, horizon_s, stride_s)`: mean over windows of the position RMSE along the
   window; stride in **seconds** (converted per piece's dt); `inf` if any window is non-finite.

**Why three errors:** correction (did the regression fit?), one-step (is one step right?), rollout
(does the simulation work?). The course showed correction error falling while rollout error levelled off.

## 7. `train.py` — fitting, with the checkpoint chosen on validation rollouts

1. `set_seed(seed)`: `random`, `numpy`, `torch`.
2. `fit_kf(u, acc_z, p)`: least squares of `m·acc_z = kf·ΣΩ²` → `kf = Σ(m·acc_z·S)/ΣS²`, `S = ΣΩ_i²`.
   TRAIN rows only (leakage rule). The reference fit gives 1.488e-6 (prior 1.5625e-6, −4.8%).
3. `fit_linear_drag(v_body, f_res)`: `np.linalg.lstsq(v_body, f_res)` returns `Dᵀ`.
4. `TrainConfig` (dataclass) + `from_dict(cfg)` reading `mlp`, `optim`, `early_stopping` from `train.yaml`.
5. `fit_network(kind, p, X, Y, x_fallback_scale, val_score, cfg, seed, min_std)`:
   - fit the `Normalizer` on `X, Y` (train rows), make float64 tensors of the scaled data;
   - `MLP([10, *hidden, 6]).double()`, Adam, a seeded `torch.Generator` for `randperm` mini-batches;
   - every `val_every` epochs (and at epoch 0): build a frozen model, `score = val_score(model)`;
     keep the best `state_dict` if `score < best·(1 − min_rel_improvement)`, else count patience;
   - stop on patience or `max_epochs`; **restore the best state**; return the model and a history dict.
   - `val_score` is passed in (a closure over the validation pieces), so `train.py` knows nothing about data.

**Tests:** `tests/test_train.py` (including: the selected epoch is the best fake validation score).

6. `fit_torque_gain(u, acc, alpha, w, p) -> (3,)`: per axis, `g = Σ τ_motor·τ_meas / Σ τ_motor²` with
   `τ_motor` from `motor_wrench(u, dataclasses.replace(p, torque_gain=np.ones(3)))` and
   `τ_meas = measured_wrench(...)[1]`. TRAIN rows only; `run.py` applies it after the kf fit.

## 7b. The torque finding (why `torque_gain` and `costa_force` exist)

The diagnosis on train flights: front/back and left/right motor-speed differences explain almost none of
the measured body torque (R² ≤ 0.05 on slow flights, even with a free per-motor mixer; no lag; no
inflow or forward-speed rotor term helped). The ideal PBM torque varies 25–130× more than the measured
torque, so PBM rollouts lose attitude within 0.25 s, and CoSTA's torque target becomes 5–12× larger
than the DDM's. Two remedies, both selected on development data only:

- **PBM `torque_gain`** (already in your `physics.py`: a `QuadParams` field, multiplied in
  `motor_wrench`): one train-fitted gain per axis, ≈ [0.28, 0.19, 0.19] on the 27 train flights.
  The PBM's 0.5 s validation error drops 0.51 → 0.17 m (position), 1.56 → 0.51 rad (attitude).
- **`costa_force`**: CoSTA with `torque_prior=False` — physics prior for the force only, the network
  supplies the whole torque. Target (`labels.targets`): residual force + full measured torque.
  Implement it as a flag in `CoSTA.wrench`, a branch in `targets`, and a `kind` in
  `train.build_network_model` / `models.load_model`.

Tests: `test_torque_gain_scales_motor_torque_only`, `test_force_only_costa_ignores_motor_torque`,
`test_costa_force_target_is_residual_force_and_full_torque`, `test_fit_torque_gain_recovers_per_axis_scaling`.

## 8. Already done for you (kept): `scripts/run.py`, `configs/train.yaml`, `tests/`

`run.py` imports exactly the names above. Once steps 1–7 pass their tests it runs unchanged:

```bash
uv run python scripts/run.py --model pbm --roles val --eval-stride 20
```

Test roles only run from a saved run: `--from-run results/<run_id> --roles test_interp test_near test_extrap`.

---

## Pitfalls the reference implementation hit (so you can recognise them)

| Symptom | Cause | Fix |
|---|---|---|
| attitude error 1e-8 for identical states | `arccos` near 1 | `atan2` form |
| `RuntimeError: mat1 and mat2 must have the same dtype` | torch default float32 vs float64 data | `.double()` on the net |
| `overflow encountered in exp` | naive numpy SiLU | `z * expit(z)` |
| CoSTA batch gives wrong shapes | `c[:3]` on `(B, 6)` | `c[..., :3]` |
| batched quaternion norm wrong | missing `keepdims=True` | add it |
| `test roles are evaluated only from a frozen run` | protocol guard in run.py | evaluate test once, at the end |
