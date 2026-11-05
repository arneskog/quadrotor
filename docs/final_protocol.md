# Final evaluation protocol (pre-registered)

Declared 2026-11-04, **before any test flight has been evaluated**. Everything below is fixed;
changing it after a test result has been seen invalidates the comparison.

## Models
| name | role in the study |
|---|---|
| `pbm` | physics only: kf and per-axis torque gain fitted on train flights (unfiltered labels) |
| `linear_drag` | PBM + D v_body, least squares on train flights |
| `costa` | PBM + learned residual wrench (main hybrid model) |
| `ddm` | black-box wrench, shares the rigid-body equations |
| `ddm_kin` | secondary baseline: black-box accelerations incl. gravity, kinematics only |

`costa_force` is **not** in the final runs: on development flights it matched `costa` once the
torque gain was in the PBM (results/pseudo_extrap_20261103-145905); it is reported as an ablation.

## Shared settings
`configs/train.yaml`, `configs/params.yaml`, `configs/splits.yaml` (variant b) as committed.
Every model: same train flights, PBM parameters, inputs (ddm_kin + gravity direction), normalization
rule, MLP 64-64 SiLU, Adam, checkpoint rule (validation rollouts, 0.5 s windows every 1 s), RK4 at
the sample period, evaluation windows (stride 10 rows, horizons 0.1 / 0.25 / 0.5 / 1.0 s) and
recorded motor speeds. Seeds 0-4 for the network models; PBM and linear drag are deterministic.

## Validation choice: torque-label low-pass (made before the final runs)
- Candidates: none, 40, 20, 10 Hz (zero-phase Butterworth on `ang acc`, per continuous piece).
- Applied to the network **training labels only**. The PBM fit and every reported metric use the
  unfiltered measurements, so all models and cutoffs are scored against the same data.
- **One cutoff shared by every network model** (changed 2026-11-05 from a per-model choice, before
  any selection result was looked at): all networks then learn from identical labels, so the model
  structure is the only difference between them, with fewer tuning choices.
- Chosen on validation flights only, by a rule that favours no model: for each model (costa, ddm,
  ddm_kin) and seed 0-2, the checkpoint score `best_val_score` (mean position RMSE along 0.5 s
  validation windows) divided by the unfiltered run of the same model and seed; median over seeds,
  then mean over the three models. The lowest mean ratio wins; a cutoff replaces "none" only if it
  is at least 1 % better (the same `min_rel_improvement` used for checkpoints).
- Selection runs: `results/sel_<model>_a<cutoff>_s<seed>`; `scripts/select_alpha.py` writes
  `results/alpha_selection.md`; the result goes into `configs/train.yaml` (`labels.alpha_lowpass_hz`).
- No sample weighting (every training row weighs the same), declared in `configs/train.yaml`.

## Order of work
1. Commit the code and configs; the final runs record that commit in `run.json`.
2. Final train/val runs: every model, seeds 0-4, all networks with the shared selected cutoff.
3. Test evaluation **once per run**, from the frozen run:
   `scripts/run.py --from-run results/<run> --roles test_interp test_near test_extrap`
   (`run.py` refuses a second test evaluation of the same run). Everything figures need is saved in
   that one pass, so no figure ever needs a re-evaluation (`errors_test.npz`, keys `<role>__<name>`;
   full list in `scripts/run.py` `evaluate_role`): per-window errors at every horizon with start speed,
   flight and piece; error-quantile curves on a 10 ms grid (all windows and per speed bin); full
   predicted and measured 1 s trajectories every 2 s of every piece; per-row state, motor speeds,
   measurements, predicted and measured force/torque; one-step errors; the training box.
   `history.json` holds the training and validation regression loss and the validation rollout score.
   A correction-scaling (alpha) sweep, if wanted, is done on validation flights only.

## Results to report
- **Primary:** position error at 0.5 s, median over windows, per test role (interp, near, extrap);
  per model the median over seeds with [min, max].
- **Secondary:** all horizons; velocity, attitude and body-rate errors; error by window-start speed
  bin; force/torque (correction) error against unfiltered labels; one-step error; fraction of test
  states outside the training box (incl. tilt).
- Development-only experiments (pseudo-extrapolation, data efficiency, ddm_kin comparison) are
  reported as such and never mixed with test numbers.

## Explicitly out of scope (future work)
- Motor-acceleration (`dmot`) input: on train flights it explains only 12-17 % of the yaw-torque
  residual, with a coefficient 5-7x below the motor inertia; yaw barely affects position.
- History inputs (NeuroBEM uses a 20-sample window): would change the rollout state.
