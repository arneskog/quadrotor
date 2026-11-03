# Quadrotor Hybrid Modeling (CoSTA) — Project Brief

Course project, NTNU "Hybrid Analysis & Modeling". Oral exam + viva covering the full course.
Goal: physics-based quadrotor model (PBM) + neural corrective source term (CoSTA) learned from
real flight data (NeuroBEM). Compare PBM vs. PBM+linear drag vs. black-box DDM vs. CoSTA.

## Stack
Python 3.11, numpy, scipy, pandas, torch, matplotlib, pyyaml, pytest. Later: pysindy.
Plain PyTorch, no PINN/ML frameworks. Everything reproducible from configs + seeds.

## Layout
```
data/raw/            NeuroBEM CSVs (git-ignored)    data/synthetic/  generated (git-ignored)
src/quadrotor/   package; modules below live here (import as quadrotor.physics etc.)
src/physics.py       f_pbm(x,u,p) -> xdot; thrust/torque from motor speeds
src/integrators.py   rk4_step, rollout(x0, u_seq, model, dt)
src/synthetic.py     "true" model = PBM + known drag; controller; trajectory generator
src/dataset.py       load NeuroBEM, frame conversions, FLIGHT-level splits
src/labels.py        residual force/torque labels
src/models.py        CoSTA MLP, DDM MLP, linear-drag baseline (least squares)
src/train.py         training, early stopping on VALIDATION ROLLOUT error
src/evaluate.py      correction / one-step / horizon-rollout errors, plots
configs/*.yaml       params, splits, seeds        tests/   pytest
notebooks/           exploration + figures        results/ metrics, figs (per run)
```

## Model
- State x = [p(3), v(3), q(4, unit quat), w(3)]; input u = motor speeds Omega(4).
- m*vdot = R(q)[0,0,kf*sum Omega_i^2] - m*g*e_z (+ F_corr rotated to world)
- J*wdot = tau(Omega) - w x Jw (+ T_corr)
- qdot = 0.5 * q ⊗ [0,w]; renormalize q after each RK4 step.
- Params (m, J, arm, kf, km, rotor layout) from NeuroBEM paper/code -> configs/params.yaml. Document sources.
- NeuroBEM frames: world z-up, body front-left-up.

## CoSTA
- Labels: F_res = m*a_meas - F_pbm(x,u) (body frame); T_res = J*alpha_meas - T_pbm(x,u).
  Always compute the residual directly (never "operator applied to error" shortcut).
- Network input: body-frame velocity, body rates, motor speeds. NO position, NO time.
- MLP ~10 -> 64 -> 64 -> 6, tanh/SiLU; normalize with TRAINING stats only.
- Rollout: RK4 with PBM + correction, driven by recorded motor speeds.
- Correction ≡ 0 must reproduce PBM bit-for-bit (test this).

## Protocol (non-negotiable)
- Split by WHOLE FLIGHTS before anything else; never random rows.
- Train/val: slow-medium flights. Test-interp: held-out flights in same speed range.
  Test-extrap: fastest flights. Report fraction of test states outside training box.
- Checkpoint selection on validation rollouts; test frozen, once.
- Quadrotor is open-loop unstable: evaluate many short windows (0.1–1 s) from measured
  states; report error vs. horizon. Also report correction error and one-step error separately.
- 5 seeds, median + range. Same data/integrator/splits for all models.

## Data facts (verified)
- 95 flights / 247 segments, 400 Hz EXCEPT 2021-02-18-16-43-54 (~164 Hz). Segments have time gaps.
- vel is BODY frame; quat (x,y,z,w) body->world; acc = body specific force (R acc - g e_z = dv_world/dt);
  dmot = d(mot)/dt. Survey: scripts/survey_flights.py -> data/flight_summary.csv.
- Split is fixed in configs/splits.yaml (train/val/test_interp <=10 m/s, test_near 10-13, test_extrap >=13,
  variant b). Rows with z < 0.5 m dropped in every role. Never change after seeing test results.
- Rollouts/filters only inside one continuous piece (dataset.continuous_pieces / rollout_windows).

## Leakage rules (examiners probe these)
- PBM params (kf, km, ...) fitted on TRAIN flights only. Mass from the paper, cited.
- Rollout network input = the MODEL's own state (v_body, w) + recorded motor speeds (+ dmot).
  Never measured velocity/acceleration inside a rollout. Labels may use the future; inputs may not.
- Normalization stats from TRAIN rows only, frozen; near-constant channels use declared physical scales.
- PBM / DDM / CoSTA share flights, inputs, normalization, checkpoint rule, seeds, RK4 dt, horizons,
  window starts and recorded motor speeds. DDM predicts the full force; CoSTA the residual.
- Torque labels: check tau_nominal vs J*alpha sign per axis on one flight before training.
- Validation choices: checkpoint, alpha low-pass cutoff (incl. none), sample weighting.

## Phases
1. PBM + data loading; sanity plot of PBM vs. measured accelerations vs. speed.
2. Synthetic benchmark (known drag) end-to-end; parity plot learned vs. true correction.
3. CoSTA on NeuroBEM.
4. Baselines + full evaluation tables/plots.
5. Extension: SINDy on learned correction and/or inverse PINN for kf/drag.

## Tests
- rk4 matches analytic solution on simple ODE; quaternion norm preserved.
- Hover with kf*sum Omega^2 = m*g -> zero acceleration.
- Synthetic: exact label in one step reproduces next reference state.
- Zero correction == PBM.

## Conventions
Type hints, docstrings with units (SI), small pure functions, no hidden global state.
Figures saved to results/<run_id>/ with config copy.
