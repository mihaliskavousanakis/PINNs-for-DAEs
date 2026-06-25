# Step-by-Step Guide to Reproducing Each Figure

This document maps every figure and result in the manuscript and supplementary information to the script(s) that generate it.

## Main manuscript

### Figure 1 — Nagumo traveling wave (Case 1)

* Folder: `case_studies/01_nagumo_forward/`
* Run: `main.m`
* Settings reproduced from the paper:
  * Two networks: `N_w(y, t)` (2 hidden layers × 20 neurons, tanh), `N_p(t)` (1 hidden layer × 5 neurons, tanh).
  * Domain `y ∈ [-30, 30]`, `t ∈ [0, 20]`.
  * Collocation: 599 spatial × 200 temporal = 119,800 PDE points; 201 algebraic-constraint points.
  * Optimizer: L-BFGS, 10,000 epochs.
* Outputs in the workspace:
  * `Loss` → Fig. 1a (total loss vs. epoch).
  * `forward(net1, …)` on a `y–t` grid → Fig. 1b (heatmap of `w(y, t)`).
  * `forward(net2, t)` → Fig. 1c (`dc/dt` vs. `t`).
  * Snapshots `w(y, t = 0.5, 2, 20)` → Fig. 1d.

### Figure 2 — Nagumo parameter inference (Case 1, inverse)

* Folder: `case_studies/01_nagumo_inverse/`
* Run: `main.m`
* Inputs:
  * `data_for_Nagumo_fifth.mat` — synthetic sensor data (`u_obs` at `x* = -10`, `t* = 2, 4, 6, 8, 10`).
  * Reference solution computed with `D = 2`.
* The vanilla-PINN comparison (Fig. 2a–b solid line with rectangles) is reproduced from `vanilla_PINN_baseline/main.m`.

### Figure 3 — 2D diffusion self-similar (Case 2)

* Folder: `case_studies/02_diffusion_2D/`
* Edit: comment out / remove the lone `return` statement before the L-BFGS loop in `main.m`.
* Run: `main.m`
* Settings:
  * `N_w(ξ, η, τ)` — 3 hidden × 40 neurons; `N_p(τ)` — 1 hidden × 5 neurons.
  * Collocation: 70,321 points in the rescaled `(ξ, η, τ)` domain `(0,4) × (0,4) × t ∈ [0, 3]`.
* Outputs map directly to Figure 3a–d.

### Figure 4 — 2D porous medium polar (Case 3)

* Folder: `case_studies/03_porous_medium_2D/`
* Run: `main.m`
* Settings:
  * `N_w(ρ, θ, τ)` — 3 hidden × 40 neurons; `N_p(τ)` — 1 hidden × 5 neurons.
  * Domain `(ρ, θ) ∈ (0, 4) × (0, π/2)`, 85,571 collocation points.
  * 30,000 L-BFGS epochs.
  * Template `T(ρ) = 1 − 2 / (1 + exp(−5(ρ − 1)))`.
  * Initial condition `w(ρ, θ, 0) = 1 / (1 + exp[−5 (2 − ρ + 0.3 cos(2θ))])` — the asymmetric form in the manuscript. The line is currently `U0 = 1 ./ (1 + exp(-5.*(2-R)))` (radially symmetric); replace with `Ri = R - 0.3*cos(2*Th); U0 = 1 ./ (1 + exp(-5.*(2 - Ri)));` to obtain Fig. 4 exactly.

### Figure 5 — Burgers (Case 4)

* Folder: `case_studies/04_burgers/`
* Run: `main.m`
* No warm start; uses 179,700 collocation points in the `y–τ` domain.
* `result_10000_for_paper.mat` is included for direct post-processing if the user prefers not to retrain.

### Figure 6 — gKdV steady-state blow-up (Case 5)

* Folder: `case_studies/05_gKdV_steady_blowup/`
* Run: `main.m`
  * For Fig. 6a (`p = 5.2`): keep `p = 5.2` in `modelLoss.m`.
  * For Fig. 6b (`p = 5.6`): set `p = 5.6` in `modelLoss.m`.
* The warm-start file used for the published runs is `initial_weights_gaussian.mat` (loaded by `main.m`).

## Supplementary Information

### Figure SI-1 — 1D diffusion self-similar (SI §1)

* Folder: `supplementary/SI1_diffusion_1D/`
* Run: `main.m` (15,000 L-BFGS epochs, no warm start). Use `result_15000_for_paper.mat` to skip retraining.

### Figure SI-2 — Axisymmetric 2D PME, 2nd-kind self-similarity (SI §2)

* Folder: `supplementary/SI2_porous_medium_axisymmetric/`
* Run: `main.m`
* Loads `result_7_new.mat` as a warm start (the script continues a previous training run). To start from scratch, comment out `load result_7_new` and use the fresh random initialization.

### Figures SI-3 to SI-6 — Baseline finite-difference / finite-element comparisons (SI §3)

* SI §3.1, Fig. SI-3 — `baselines/nagumo_FD/main_nagumo.m`
* SI §3.2, Fig. SI-4 — `baselines/porous_medium_axisymmetric_FD/main.m`
* SI §3.3, Fig. SI-5 — `baselines/porous_medium_2D_polar_FE/*.mph` (open in COMSOL).
* SI §3.4, Fig. SI-6 — `baselines/burgers_FD/main.m`

### Figure SI-7 — Robustness studies (SI §4)

* §4.1 (random initialization, Burgers): rerun `case_studies/04_burgers/main.m` after setting `rng(k)` for `k = 1, …, 5` at the top of the script.
* §4.2 (KdV pretraining sensitivity): rerun `case_studies/05_gKdV_steady_blowup/main.m`, each time replacing the line `load initial_weights_gaussian` with one of the alternative files (`init_weights_TW.mat`, `init_weights_gaussian_dev_05.mat`, `init_weights_gaussian_dev_2.mat`).

## Common plotting recipes

```matlab
% 2D field — heatmap of the rescaled solution
[T, Y] = ndgrid(linspace(t_min, t_max, Nt), linspace(y_min, y_max, Ny));
W = extractdata(forward(net1, dlarray([T(:)'; Y(:)'], "CB")));
W = reshape(W, size(T));
figure; pcolor(T, Y, W); shading flat; colorbar;
xlabel('\tau'); ylabel('y');

% Scaling rates from N_p
P = extractdata(forward(net2, dlarray(linspace(t_min, t_max, 200), "CB")));
figure; plot(linspace(t_min, t_max, 200), P(1,:), '-', linspace(t_min, t_max, 200), P(2,:), '--');
legend('\partial_\tau A / A', '\partial_\tau B / B');

% Convergence
figure; semilogy(Loss); xlabel('epoch'); ylabel('\mathcal{E}_{total}');
```
