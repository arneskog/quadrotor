# TTK29-2026 — course notes for the quadrotor CoSTA project

Course: "Hybrid Analysis and Modeling for Digital Twins", Adil Rasheed, NTNU ITK.
Source: `TTK29-2026/` (git clone, ignored). Slides so far: L00 intro, L01 PBM, L02 DDM, L03 PINN,
L04 CoSTA, L05 compressed sensing + optimal sensor placement. Code: `TTK29-2026/python/<NN-module>/`.
Oral exam + viva covers the WHOLE course, not only CoSTA. Summary written 2026-11-03; re-check the
repo for new slides (`git -C TTK29-2026 pull`).

## Course frame (L00) — vocabulary examiners will use

- Digital twin = virtual representation of an asset/process, enabled by data and simulators, for
  real-time prediction, optimisation, monitoring, control, decision-making. "Real-time" ⇒ cheap models;
  "informed" ⇒ state the uncertainty.
- Capability scale 0–5: standalone, descriptive, diagnostic, **predictive (3)**, prescriptive, autonomous.
  This project is a level-3 building block (forecast state from a model + data).
- PBM vs DDM: PBM interpretable, extrapolates as far as its assumptions, needs little data, fails by
  missing physics/cost. DDM accurate in-distribution, fast, data-hungry, fails silently under shift.
  Thesis: they fail in complementary ways ⇒ hybrid analysis and modeling (HAM).
- "Squeeze the residual": physics first, then interpretable DDM, then black box, stop when the residual
  is noise. Resolved ⊂ modelled ⊂ observed ⊂ full physics.
- Hybrid *analysis* = data dominant, physics interprets. Hybrid *modelling* = physics dominant, data
  supplies the missing term/parameter/closure (CoSTA, SINDy, PINN, ROM, DMD, CS, DeepONet, FNO).
- Course map (31 modules): Intro 00–02 · I multivariate 03–08 (LS, MLR, PCA, PCR, PLS, ICA) ·
  II nonlinear 09–12 · III time series 13–14 · IV deep learning 15–18 · V hybrid 19–26
  (ROM, CoSTA, SINDy, PINN, DMD, CS, DeepONet, FNO) · VI control 27–30 (MPC, Koopman, RL, safe RL).
- Course notation: scalar italic, vector bold lowercase (θ for network params), matrix bold capital.

## L01 PBM — recipe the course expects

- Three-part study for every model: **derive** (balance laws + closures; closures = "amber" uncertain
  terms) → **discretize** (scheme + stability limit) → **results** (find where it breaks).
- Vessel benchmark (closest analogue to the quadrotor): Newton–Euler in the body frame,
  M ν̇ + C(ν)ν + D(ν)ν = τ, η̇ = R(ψ)ν (Fossen form). Coriolis terms appear because the frame rotates
  — same reason as our w × Jw.
- Stability, ODE side: eigenvalues of the linearisation in the left half-plane. Quadrotor: open-loop
  UNSTABLE (attitude errors tilt thrust) ⇒ short rollout windows; say this explicitly.
- RK4 lecture claims: local error O(Δt⁵); fitted global slopes Euler≈1, RK2≈2, RK4≈4; step chosen
  for accuracy, not stability, for non-stiff ODEs. A fine RK4 run is "honest ground truth".
- Stiffness/CFL/von Neumann: explicit step caps (FTCS r ≤ ½, CFL C ≤ 1); stiff ⇒ implicit (backward Euler).

## L02 DDM — what the black-box baseline must show

- Supervised loop: data → model fθ → cost (fit + λR) → optimizer; train/val/test, golden rule: test
  never influences any choice.
- Defaults the course uses: Adam, lr 1e-3–2e-3 with decay, standardised inputs/outputs, smooth activations,
  Glorot init, early stopping = "cheapest, most effective regulariser".
- Regularisation: L2 (ridge, shrink), L1 (lasso, exact zeros), dropout, augmentation; "physics is the
  ultimate regulariser".
- UQ: aleatoric vs epistemic; deep ensembles, MC dropout, heteroscedastic head; calibration. Epistemic
  uncertainty should blow up under extrapolation. (Our 5 seeds = a poor-man's ensemble.)
- Vessel DDM lesson: one-step R² ≈ 1 but rollout drifts; OOD cliff; more data fixes interpolation,
  not extrapolation; fine-tuning on k-step rollouts softens drift, does not cure it.

## L03 PINN — needed for the phase-5 extension and the exam

- Loss = physics residual + IC + BC + data, each averaged over its own point set; residual via autodiff
  (`create_graph=True` for higher derivatives). Smooth activations (tanh/sin/GELU); ReLU kills u_xx.
- Forward vs **inverse** (λ trained with θ). Vessel inverse PINN recovers damping; **identifiability needs
  excitation** (slow maneuvers leave linear/quadratic drag correlated). Analogue for us: identifying kf
  vs drag needs fast flights.
- Failure modes: loss imbalance (gradient-norm balancing), spectral bias (Fourier features/SIREN),
  causality (causal weights, time-marching windows/XPINN slabs), stiffness (non-dimensionalise).
  "Low loss ≠ low error."
- Variants: cPINN (interfaces, flux continuity), XPINN (space-time slabs), vPINN (weak form), ensembles.
- Adam → L-BFGS refinement. For ODE system ID the course recommends a custom PyTorch loop, not a framework.
- Hand-off quote: "if no α reconciles physics and data, the equation is missing a term — the setting of CoSTA."

## L04 CoSTA — the core of this project

Definitions (use the lecture's words in the exam):
- Three objects: reference u_ref; runnable model L̃ũ = f̃ (in practice the solver step);
  corrected model L̃u = f̃ + σ.
- **Exact corrective source σ* = L̃ u_ref − f̃** — the residual of the reference in the runnable model.
  With σ = σ* and matching IC/BC the reference solves the corrected problem.
- Never use the "operator applied to the error" shortcut L̃(u_ref − ũ): only valid for linear L̃.
- An interior source cannot repair wrong boundary data.
- Four error cases: (1) source/BC, (2) parameter/operator, (3) truncation only, (4) both.
  σ* = (L̃ − L)u_ref [operator/parameter] + (f − f̃) [source] + (L̃_num − L̃)u_ref [truncation].
  Inputs must expose what the correction depends on; a truncation correction belongs to its grid/step.
- Our quadrotor labels F_res = m a_meas − F_pbm are derivative-based (like the vessel), so they contain
  missing aerodynamics + parameter error (kf, km, J) but essentially no RK4 truncation. Mention this.

Using the correction:
- Zero-correction property: σθ ≡ 0 ⇒ exactly one PBM step. Does NOT imply "never worse than PBM",
  "switching off undoes the error", or "stability carries over".
- Two loops: current-state (course notebooks, one solve/step — ours) vs predictor-corrector
  (Blakseth et al., two solves/step).
- Scaling the correction (α fθ) is "not a safety dial": α = 0.1–0.4 was worse than the PBM; a partial
  correction is a different model.

Protocol (the examiners' checklist, slide 44):
- Labels may use the future; inputs may not. In a rollout the network sees the model's own state.
- Split whole trajectories first, then extract rows (random row split leaks).
- Normalise with training rows only; near-constant channel ⇒ declared physical floor, not std≈0
  (vessel: an unfloored zero-std channel gave inputs of −6×10¹⁰).
- Validate by rollout and restore the best checkpoint; test frozen and recursively, once.
- Report spread over declared seeds; record failed (non-finite) runs.
- Three errors, reported separately: **correction** (σθ vs label on reference states), **one-step**,
  **rollout**. Correction MSE kept falling while validation rollout levelled off ⇒ select on rollouts.
- Interpolation vs extrapolation is about **states**, not parameters: report fraction of test states
  outside the training box (heat: a = 0.35 is "extrapolation in a" but its states are inside the box).
- Checklist: which trajectories in which role; how labels are built (scheme inversion or derivatives,
  filter); what the input may contain at prediction time; how normalisation is fitted; what selects
  checkpoint/settings; frozen recursive test; metric and units; test states inside training range;
  seed spread and failures.

Benchmarks and numbers worth quoting:
- Heat (manufactured, missing reaction term): rollout RMSE PBM 0.68 vs CoSTA 1.3e-3 (a = 2.25);
  in-range 350–500× better; extrapolation degrades with T outside the trained range.
- Vessel (hidden drag d = −[3u|u|, 8v|v|, 2r|r|]): position RMSE on held-out zig-zag PBM 5.8 m,
  CoSTA 0.13 m; extrapolation zig-zag PBM 11.9 m, **DDM 24.3 m (worse than PBM)**, CoSTA 0.77 m.
  CoSTA MLP 3→32→32→3, DDM 6→64→64→3; DDM still improving when its budget ran out (stated honestly).
- Wind turbine (synthetic erosion): transfer above rated needs training data in the new regime AND the
  input (pitch β) the missing torque depends on — validation would already have rejected the no-β variant.
- Noisy labels: validation picked NO smoothing (moving average biases labels; zero-mean noise averages out).
  ⇒ our alpha low-pass cutoff must be a validation choice, including "none".
- Should the net see x, t? Validation picked stencil-only; past the training horizon the time input was 50× worse.
  ⇒ justifies NO position, NO time in our inputs.
- Stability: corrected step bounded if network Lipschitz L_θ < physical damping rate λ_min. Free-form sources
  break conservation; a missing flux needs a flux-form correction.
- Correction is data for discovery (→ SINDy, module 21): a structured persistent correction suggests a genuine
  missing term, but the label also holds truncation and different mechanisms can give the same discrepancy:
  a discovered term is a hypothesis to test on new data.
- Limits: needs reference data and derivatives; inputs must expose dependencies; extrapolation is a guess;
  rollouts can drift; same solver/grid/step in training and deployment.

"Check your understanding" (likely exam style):
1. Correction MSE 1e-6 on a random 20 % of rows → correction error, leaky random-row split.
2. Vessel correction fed the reference velocity every step → one-step, not rollout.
3. New maneuver whose (u,v,r) all lie in the training range → interpolation.
4. "Zero correction is the PBM" ⇒ never worse than PBM? No.

## L05 Compressed sensing and optimal sensor placement (exam coverage)

- x = Ψs, s k-sparse; y = Cx = Φs with Φ = CΨ; needs sparsity AND incoherence (μ, RIP).
- ℓ0 is NP-hard → basis pursuit (ℓ1, convex; identical to ℓ0 under RIP). ℓ1 gives exact zeros, ℓ2 shrinks.
- Phase transition m ≈ 1.47 k log(n/k) (measured); budget 2–3× the 50 % line; below it ℓ1 is confidently wrong.
- Solvers: BP, BPDN, lasso, ISTA/FISTA (soft-threshold, 1/k vs 1/k²), OMP. Debias: lasso to select, LS to refit.
- Use when signal truly sparse, readings expensive/noisy (multiplex advantage), or measurements not pixels (MRI).
- Sensor placement: POD basis Ψ_r from SVD of snapshots; gappy POD â = Θ⁺y, Θ = CΨ_r; D-optimal = max det(ΘᵀΘ);
  greedy pivoted QR on Ψ_rᵀ (`scipy.linalg.qr(Psi.T, pivoting=True)`). Placement matters most when m ≈ r.
- Links to this project: ℓ1/sparsity is the engine of SINDy (phase 5); "check compressibility first" mirrors
  "check the residual is structured before discovering a law".

## Course code conventions worth matching (`TTK29-2026/python/`)

- `pyutils/nets.py`: `MLP(layers)` = tanh hidden layers, linear output, Xavier-normal weights, zero bias.
- CoSTA notebooks: `torch.set_default_dtype(torch.float64)`; full-batch Adam, lr 2e-3, StepLR halving every
  2000 steps; validation rollout every 250 steps, patience 6, must beat best by 1 %; non-finite rollout = +inf;
  restore best `state_dict`; rollouts run in numpy with frozen weights (`numpy_weights` + `forward`) for speed,
  with an assert that numpy forward == torch.
- Protocol asserts in the notebook: no maneuver in two roles; numpy forward matches torch; evaluation leaves
  weights unchanged; normalised inputs finite and |z| < 10; α = 0 reproduces PBM bit for bit.
- Labels: central differences within one trajectory, dropping samples whose stencil crosses a control switch
  (analogue for us: never across a NeuroBEM time gap; use `continuous_pieces`).
- Every notebook ends with asserts enforcing the numbers on the slides; results saved to `data/*.json`.
- Environment: conda `environment.yml`, Python 3.11, torch + pysindy via pip.
- Relevant notebooks: `20-corrective-source-term/notebooks/costa_vessel_ode.py` (closest to us),
  `costa_heat_pde.py` (full protocol), `costa_error_cases.py`, `costa_wind_turbine.py`;
  `22-physics-informed-nn/notebooks/pinn_vessel_ode.py` (inverse ODE PINN), `pinn_ensemble_vessel.py`.
