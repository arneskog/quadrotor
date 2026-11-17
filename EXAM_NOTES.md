# Exam notes: what the notebooks add beyond the slides

Source: `TTK29-2026/` as of 2026-11-17 (slides L00–L07; notebooks in `python/00, 01, 02, 20, 21, 22, 24`).
The slides already quote most notebook numbers. This file lists only what is **in the notebooks and missing or
only hinted at on the slides**: the reasoning, derivations and caveats an examiner can ask about. Each item is
marked ★ if it is likely viva material.

No slides or notebooks exist for modules 03–19, 23, 25 (DeepONet) or 26 (FNO). They appear only on the course map
and in the CoSTA "one rod, six methods" table. Know what they are in one sentence each, not more.

---

## L00 Introduction: `intro_three_paradigms`

The whole course in one toy problem: surge coast-down `m u̇ = −d1 u − d2 u|u|`, true `d2 = 2.0`, PBM `d2 = 1.2`.
Train on t ∈ [0, 4], predict t ∈ [4, 8].

- ★ **PBM error is bias, not noise.** It is wrong even inside the data window, and more data does not move it. Only
  a better coefficient does.
- ★ **The hybrid is a *structured* correction** `g(u) = c·u|u|`, found with one least-squares scalar:
  `c = ⟨r, u|u|⟩ / ⟨u|u|, u|u|⟩` = −0.0815 (analytic −0.08). "Structure is what extrapolates": a free network
  correction would fall off the same cliff as the DDM.
- ★ **Do not finite-difference noisy data.** The notebook gets `u̇_data` by autodiff of the smooth DDM fit, not by
  differencing the noisy samples. This idea returns in modules 20 and 21 (weak form).
- The hybrid beats both pure models by more than 3× out of sample (asserted).

## L01 PBM

**Vessel (`vessel_ode`)**
- ★ Numbers to know: `M = diag(55, 80, 30)` = rigid body (50, 50, 20) + added mass (−X_u̇ = 5, −Y_v̇ = 30,
  −N_ṙ = 10). `D(0) = diag(10, 40, 15)`.
- ★ **Linearisation at rest:** the quadratic drag vanishes and the Coriolis term is second order, so
  `ν̇ ≈ −M⁻¹D(0)ν`. It is diagonal, so the eigenvalues are just `−d_ii/m_ii` = −0.182, −0.5, −0.5. Surge is the
  slowest mode (τ ≈ 5.5 s) because it is the least damped.
- ★ **C(ν) is skew-symmetric** (C13 = −m22·v, C23 = m11·u). Physically, Coriolis forces do no work. This is the
  basis of the SINDy energy check in L06.
- Coast-down: early decay is faster than `e^{λt}` because the quadratic drag adds damping at high speed.
- Turning circle: the final heading of 144 rad is the *unwrapped* heading (~23 turns).

**Heat (`heat_diffusion`, `heat_cable`)**
- **Maximum principle:** with no source the interior stays between the coldest and hottest wall value.
- τ = L²/α = 10⁴ s. The stable FTCS step is 0.625 s (with a 0.9 safety factor), so steady state takes about 10⁴ steps.
- ★ **Cable:** finite volumes in r with a **harmonic-mean conductivity at the copper/XLPE interface** (keeps the
  interface flux continuous) and backward Euler (unconditionally stable). The explicit limit would be 2.7×10⁻⁴ s,
  for a process that takes hours. The solver is validated against the analytic steady state (34.83 vs 35.01 °C).
  Time constants: t63 ≈ 70 min, t95 ≈ 3.7 h.
- Copper conducts ~1400× better than XLPE, so the core is isothermal and **the insulation is the thermal
  bottleneck**.

**Wind turbine (`wind_turbine`)**
- Heier fit `Cp(λ, β)` (formula in the notebook). `K = ½ρAR³Cp_max/λ_opt³`.
- ★ **Why MPPT works:** the steady state of `Jω̇ = T_aero − Kω²` lies exactly at λ_opt, so the law tracks the optimum
  automatically.
- Pitch PI (Kp = 18, Ki = 6) with anti-windup, pitch clamped to [0, 40]°. Semi-implicit Euler, Δt = 0.05 s.
  Ornstein–Uhlenbeck wind.
- ±10 % in Cp gives about ±6 % in mean energy (2.68 / 2.86 / 3.02 MW). Above rated all curves collapse onto 5 MW,
  because region III sheds the excess.

**Navier–Stokes (`navier_stokes_cavity`, `navier_stokes_vortex`)**
- ★ **Pressure is a Lagrange multiplier** for incompressibility. It has no time derivative and is fixed by the
  constraint.
- Chorin projection with 50 Jacobi sweeps. Pressure BCs: Neumann on the walls, p = 0 reference at the lid.
- ★ **Two step limits at once:** `Δt = min(0.2Δx²/ν, 0.5Δx/U)` (diffusive and CFL).
- ★ **Cell Reynolds number** `UΔx/ν = Re/(nx−1) ≲ 10` for central differences of the convection term. That is why
  Re = 1000 needs at least 81² points.
- Validation signatures: max speed = lid speed; corner pressure spikes (+0.4 downstream, −0.24 upstream).
- Vorticity–streamfunction form `∇²ψ = −ω`, solved spectrally (`ψ̂ = ω̂/|k|²`) and marched with RK4. **Circulation is
  conserved exactly** by the periodic spectral scheme, while the peak vorticity decays through viscosity (a
  conserved quantity next to a non-conserved one). RK4 diffusive limit: `ν k_max² Δt ≲ 2.8`.

## L02 DDM

- ★ **A spatial split turns validation into extrapolation.** Train on x ≤ 0.6 and validate on x > 0.6, and the
  "best" model becomes a straight line (validation MSE up to 10⁷ at degree 8). "The lesson is about the split, not
  the model."
- ★ **Gradient-descent stability:** the learning rate must satisfy `η < 2/λ_max(Hessian)` (0.796 here). The bowl is
  elongated because the features are correlated, so the gradient points across the valley. SGD jitters near the
  minimum, and a decaying learning rate removes the jitter.
- An early-stopping run: best validation at epoch 1969, stop at 2470 (patience 500), test once ≈ the noise floor
  σ² = 0.0225.
- ★ **Ensemble vs MC dropout** (`ddm_uncertainty`): the ensemble's spread is 13× larger in the data gap than on the
  data; MC dropout's is only 1.4×. Dropout noise inflates the band *everywhere*. Hence the course uses ensembles to
  catch extrapolation.
- ★ **Vessel DDM split by whole maneuver.** Consecutive steps are near-copies, so a split by step leaks.
  Standardisation uses training pairs only. Pose is integrated with the *known* kinematics. Numbers: one-step
  R² ≈ 0.996, final drift 2.74 m in range, 77 m on the zig-zag (~30×). With more data the in-range error falls
  from 4.3 to 1.2 m, while the out-of-range error stays at 23–39 m. k = 12 rollout fine-tuning gives 0.89 → 0.78 m
  (−12 %).
- Heat surrogate: the split is along the α axis. Out of range, the surrogate **breaks physics it was never told**:
  boundary-condition violation 0.47 and asymmetry 0.41 on a unit scale. A 4-member ensemble's spread grows 17× out of
  band, so it flags the cliff without seeing the truth. ★ The speed-up is only 4× (1.3 ms vs 6 ms): on a small grid
  the cost argument is modest.
- NS surrogate: the lid BC u = 1 is predicted as 0.10, and the error is ~500× larger above the sensor line.
- SCADA power curve (`ddm_wind_power_curve`), real-world practice:
  - the IEC **method of bins** (0.5 m/s bins);
  - 10-min averaging *lifts* region-II points (the cubic is convex, i.e. Jensen) and lowers near-rated points;
  - ~6 % curtailment outliers;
  - ★ an MLP on (v, TI, ρ) beats binning by ~8 %: feature engineering, not capacity;
  - with 50 records, validation selects a width-1 network;
  - trained below 12 m/s, it bends *down* (3.98 vs 5 MW). "Coverage, not cleverness."

## L03 PINN

- ★ **Finite differences have a V-shaped error** (truncation ∝ h², rounding ∝ ε/h). The best possible error is about
  ε^{2/3} for f′ and ε^{1/2} for f″, at a step you would have to know in advance. AD is exact up to rounding.
- ★ **Manufactured check:** apply your residual code to the exact solution, and it must return zero.
- ★ **The output bias gets zero gradient from the residual.** A constant shift changes neither u_t nor u_xx, so the
  residual alone does not determine a unique solution. The IC/BC terms fix the level.
- ★ **What a ReLU heat PINN actually solves:** u_xx ≡ 0, so the residual reduces to u_t, and the "solution" is the
  initial profile frozen in time. The weak form (vPINN) makes ReLU usable again.
- Autograd pitfalls:
  - `grad_outputs = ones` gives the per-sample derivative only if samples do not interact (batch norm breaks it);
  - for a vector output it sums the components;
  - a Laplacian needs one backward pass per coordinate (applying all-ones twice gives Hessian row sums);
  - `retain_graph` is not `create_graph`.
- L-BFGS needs a **fixed** collocation set, because its line search needs a deterministic loss. The fair comparison
  with Adam uses an equal number of loss-and-gradient evaluations.
- The inverse heat PINN uses only interior data, with **no IC or BC**: physics fills the space between sensors.
- ★ **Grey-box PINN** (vessel §5): a known form `c·u|u|` with trainable c. Constant thrust keeps the term excited.
  If the *form* is unknown, this becomes CoSTA.
- ★ **Ensemble PINN:** the histogram shows a second basin near d2 ≈ 1, so ensembles expose local minima that a single
  run hides. The band reflects run-to-run variability, **not model-form bias**: a wrong equation gives a narrow band
  around a wrong value.
- cPINN: why one network gives u = 1 − x. With piecewise-constant k the strong residual is k·u″ = 0 on each side; the
  jump enters only through k′ (a delta at γ), which no collocation point ever sees. Exact solution: thermal
  resistance `R = γ/k1 + (1−γ)/k2`.
- The wake PDE `U ∂x d = νt ∂yy d` is the heat equation with x playing the role of time, so the analytic Gaussian
  wake serves as ground truth.
- Parameter-conditioned PINN `u(x, t; c)` is the alternative to DeepONet for a family of problems.

## L04 CoSTA (most is already in NOTES.md)

- ★ The heat label is **linear in the stencil**, because `T^{n+1} − 1 = e^{−Δt}(T^n − 1)` for this family. A network
  is more than needed, which makes it a clean benchmark: any error comes from training, coverage and feedback.
- ★ **Difference basis** (heat rod): feed `[T_i, slope, curvature]` instead of the raw `[T_{i−1}, T_i, T_{i+1}]`.
  It is an invertible linear change (no information added), but conditioning matters: the three temperatures agree to
  3 digits while the label scales with their second difference (~10⁻³). This was chosen on validation.
- ★ **Why CoSTA is data-efficient:** the local per-step formulation turns 60 trajectories into 1.5×10⁶ training rows.
  A fixed-output map (40 sensors → final profile) sees only 60 pairs and is no better than the wrong PBM.
- Bump failure explained in *input space*: the curvature at the bump peak (~625) exceeds every training stencil.
- ★ Case 2: "a wrong parameter is often better fixed by **estimating the parameter** than by learning its
  consequences." Adding x helps but does not remove the error, because the label depends on level n+1 and a level-n
  input only approximates it.
- With a single-mode reference, the current-state and predictor loops are indistinguishable (the predictor is a
  scaled copy of the state, and normalisation removes the scale).
- ★ Vessel: **replayed recorded controls**, so the test measures *prediction* error, not closed-loop performance. The
  hull is directionally unstable, so small turning differences change the whole heading history. Heading is reported
  separately and wrapped. The DDM extrapolates in the *controls* too (its input includes τ). CoSTA improves on the PBM
  by 17× and 44× in range, and 5× and 16× out of range.
- ★ **Label noise from differencing:** velocity noise s gives acceleration noise ≈ s/(√2·Δt). A moving average
  lowers it but adds **bias** (the window-average acceleration). The regression averages zero-mean noise but cannot
  average bias, so validation chose no smoothing.
- ★ "Forces that depend on time, position or heading (currents, wind, waves) need those inputs." This is the
  counterpart of the quadrotor's no-position / no-time rule: say why the quadrotor's aerodynamics are a function of
  body velocity, rates and rotor speeds.
- Wind CoSTA: **closed-loop prediction** (each model runs its own controller), unlike the vessel replay. The label
  equals the true deficit only because the synthetic reference uses the *same* explicit Euler step; real data would
  add discretisation and noise. Label noise ≈ √2·J·s/Δt ≈ 546 kN·m per rad/s. "Same mean wind ≠ same states": a
  gust put 28 % of one test case outside the training box.

## L05 Compressed sensing and sensor placement

- The phase transition has a name: **Donoho–Tanner**.
- ★ **Why ISTA and FISTA beat their worst-case rates:** once the support stops changing, the lasso is smooth and
  strongly convex on that support, and both converge *linearly*. FISTA finds the support 5.3× sooner (iteration 118
  vs 620), and afterwards the linear rates are similar. "Worst-case rates buy the guarantee; momentum buys the
  support." FISTA is non-monotone.
- λ is set as a fraction of `‖Φᵀy‖∞` (at or above that, s = 0).
- Linear interpolation fails below Nyquist because it is a **low-pass smoother**: it cannot produce the 337 Hz tone.
- A global DCT prior is weaker than *local* smoothness when the measurements are pixels of a smooth image. That is
  why interpolation wins on the portrait.
- ★ **POD details:**
  - subtract the mean field first;
  - with snapshots as rows, the spatial modes are the *right* singular vectors;
  - choose r by energy fraction (≥ 0.99999 → r = 13);
  - POD is the optimal r-dimensional basis for average projection error;
  - **the projection error of the test field onto Ψ_r is a floor no sensor set can beat**.
- QR sensors sit where the modes differ most (the meandering wake), not in the uniform inflow. A bad random set is
  catastrophic, not just mediocre (cond 660 vs 3.3).

## L06 Equation discovery (not yet in NOTES.md)

- ★ **Threshold rule:** normalise columns *and* target so λ is dimensionless, then take the **sparsest model on the
  plateau** (fit error within 1.5× of dense least squares), in the middle of that λ range.
- ★ **Identifiability on the rod:** a single sine mode has `T_xx = −π²T` exactly. The columns are collinear
  (cond ~10¹³), so α is not identifiable. Several modes decaying at different rates are needed.
  This is the same lesson as PINN excitation (slow coast-down) and the turbine dither: **the data must excite every
  term you want to tell apart.**
- Even with clean data, a doubled `np.gradient` under-resolves the high modes' curvature (relative error
  (kπΔx)²/3, 1.3 % at k = 8). Dense least squares then adds a spurious +0.02·T, which the threshold removes.
- Noise amplification on the rod: T_xx by 1/(4Δx²) ≈ 4×10³, T_t by 1/(2Δt) = 250. So the regressor has SNR < 1,
  and the coefficient is **attenuated**, not just noisy.
- ★ **Hidden state:** remedies change the state, not the library: measure it, use a delay embedding (**Takens'
  theorem**), or use a SINDy autoencoder.
- ★ **Energy check derivation:** κ = 0 follows from C(ν) being skew-symmetric. The vessel's strong drag keeps
  energy-violating models bounded; on lightly damped systems the same violation makes them blow up (hence
  constrained or trapping SINDy).
- Detective assignment: the residual correlated more with v³ than with the true |v|v. **Residual correlation can
  mislead; the error floor is the decisive test.**
- ★ **Turbine closed-loop trap, the full story:** the MPPT-identified model reproduces MPPT data to 10⁻⁶ because its
  wrong J and wrong Cp *cancel exactly* on the manifold T_gen = Kω². Change the gain by 40 % and it is >10 % off. A
  twin used to *redesign the controller* must be identified from perturbed-controller data. The drivetrain is heavily
  filtered (J is huge), so noise is not the hard part; excitation is.
- SR3: closed form `Ξ = (ΘᵀΘ + I/ν)⁻¹(ΘᵀẊ + W/ν)`, hard threshold √(2λν). It has convergence guarantees that
  STLSQ's loop lacks, and it is how PySINDy implements constrained and trapping SINDy.
- Constrained SINDy can make things *worse* on a wrong support (it moves the error elsewhere).
- Bayesian SINDy: σ̂² = RSS/(m − k). The full version uses spike-and-slab or horseshoe priors. Weak-form rows are
  nearly independent, so its posterior is more honest.
- SINDy-PI: the implicit library has a **two-dimensional** null space (the relation and x times the relation), so drop
  common factors.
- Symbolic regression: without linear scaling the drag law is never found (the tree would have to carry the −3
  itself). Without the parsimony penalty the population bloats with *introns* (subtrees that do not change the
  output).

## L07 Agentic workflow

A tooling walkthrough of the 11-stage research pipeline in `agentic-research-toolkit/`. Unlikely to be examined.

---

## ODE or PDE application: recommendation

**Pick the ODE application, the surface vessel.**

1. **It runs through every hybrid lecture you will be asked about**, with numbers at each stage:
   - PBM: Fossen form, eigenvalues −0.182 / −0.5 / −0.5, RK4 orders, amber coefficients.
   - DDM: R² 0.996 one step, but 2.74 m drift and a 77 m zig-zag cliff.
   - PINN: hard IC, d_u 25 → 9.0, excitation, 12 time-marching windows.
   - CoSTA: 5.8 → 0.13 m; DDM 24.3 m worse than PBM.
   - SINDy: 12 of 39 terms, weak form to 10 % noise, closing the loop to −3u|u|, energy check κ.
   The heat thread is equally complete, but the vessel is the one you already understand through the project.
2. **It is your project in miniature:** Newton–Euler in the body frame, Coriolis from the rotating frame (the vessel's
   C(ν)ν is your ω × Jω), RK4, residual-force labels from differentiated velocities, inputs = body velocity only,
   split by maneuver, rollout validation. Every vessel question is one you can answer from first-hand experience,
   and the project talk and the application answer reinforce each other.
3. **Less numerics to defend.** A PDE application drags in FTCS, von Neumann, CFL, the Poisson solve, stencils as
   network inputs and the truncation part of the label (Case 3). The ODE side needs eigenvalue stability and RK4
   order, and you already know both.

**But still know the heat rod.** The general questions (minutes 7–18) can still ask about von Neumann/CFL, the four
error cases, "is a = 0.35 extrapolation?" and the "one rod, six methods" table. Those are PDE-only.

**Where the vessel story is thin:** CS/OSP uses the wind wake, not the vessel. Be ready to say how sensor placement
would apply to a twin, e.g. which states to measure on the hull, or why the hidden-sway SINDy failure is an
observability problem.
