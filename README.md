# PINNs for DAEs — Symmetry-Driven PDE Dynamics

Reproducibility repository for the paper:

> Kavousanakis, M. E., Fabiani, G., Georgiou, A., Siettos, C., Kevrekidis, P. G., and Kevrekidis, I. G.
> *"Going with the Flow to Solve for Symmetry-Driven PDE Dynamics with Physics-Informed Neural Networks."*
> Nature Communications 2026

This repository contains the MATLAB code and warm-start data required to reproduce every case study reported in the manuscript and Supplementary Information.

## Overview

The code implements a Physics-Informed Neural Network (PINN) framework that combines a dynamic renormalization / moving-frame ansatz with a PINN loss. The transformed problem becomes an index-2 differential-algebraic-equation (DAE) system in rescaled (co-moving/co-scaling) coordinates. Each case study trains two networks jointly:

* `N_w(y, tau)` — approximates the rescaled (invariant) profile,
* `N_p(tau)`   — approximates the time-dependent symmetry parameters (wave speed and/or scaling rates).

Training uses MATLAB's `dlnetwork` / `dlarray` automatic-differentiation framework with the L-BFGS optimizer (`lbfgsupdate`).

## Highlights

⚡ Physics-Informed Neural Networks in dynamically rescaled coordinates for solving symmetry-driven PDE dynamics

🧠 Reformulation of PDEs into index-2 differential-algebraic equations (DAEs) through moving-frame and dynamic renormalization techniques

🌊 Simultaneous learning of the solution profile and the symmetry evolution (translations, scalings, and blow-up rates) using two coupled neural networks

🎯 Accurate treatment of traveling waves, self-similar solutions, and steady-state blow-up dynamics within a unified PINN framework

📈 Demonstrated across challenging nonlinear PDEs, including Nagumo, Burgers, porous-medium, diffusion, and generalized KdV equations

🔍 Enables both forward simulations and inverse parameter inference directly in symmetry-adapted coordinates

🔬 Includes comparisons with vanilla PINNs and classical finite-difference / finite-element baselines

💻 Fully reproducible MATLAB implementation with all case studies, pretrained models, and scripts used to generate the figures from the paper

## Abstract of the paper
We address a computational framework that integrates symmetry reduction into Physics-Informed Neural Networks (PINNs) for analyzing symmetry-driven dynamics in nonlinear partial differential equations (PDEs).
Using an auxiliary network to learn time-dependent transformations, we render the symmetry-invariant solutions stationary or slowly varying in rescaled coordinates while simultaneously inferring the symmetry parameters (e.g., wave speed, scaling rates).
This yields a modified evolution equation coupled with algebraic constraints on symmetry parameters, producing index-2 differential-algebraic equation (DAE) systems.
While conventional standard PDE/ODE solvers struggle or even fail with such high-index DAEs, we employ PINNs as an alternative approach that naturally unifies PDE residuals and algebraic constraints in a single loss function.
This allows simultaneous inference of the invariant solutions and the transformation properties without large domains, mesh adaptivity, or front tracking.
Beyond forward simulation, our framework also enables robust parameter inference from sparse, spatially offset data where vanilla PINNs fail.
Our numerical demonstrations include, among others, the {2D} porous medium, {the generalized Korteweg-de Vries} and Burgers PDE, showcasing our proposed approach as a powerful tool {for the solution of both the forward and inverse problems} for index-2 DAEs arising in nonlinear wave and scaling dynamics.

## Computational Environment

* MATLAB R2023a or later (R2024a recommended).
* Required toolboxes:
  * Deep Learning Toolbox (provides `dlnetwork`, `dlarray`, `lbfgsupdate`, `dlfeval`, layer constructors).
  * Statistics and Machine Learning Toolbox (used by some auxiliary routines).
* Reference hardware (from the manuscript): 11th-Gen Intel(R) Core(TM) i7-11800H @ 2.30 GHz, 32 GB RAM, Windows 11. All training is CPU-only — no GPU is required.
* The finite-element baselines in `baselines/porous_medium_2D_polar_FE/` and the reference models in `data/from_COMSOL/` are COMSOL Multiphysics `.mph` files (COMSOL 6.x recommended). A COMSOL license is required to open them, but the PINN reproducibility itself does not depend on them.

No Python dependencies are required.

## Repository Layout

```
PINNs_For_DAEs/
├── README.md
├── case_studies/                            # Main-text case studies (Sec. "Results")
│   ├── 01_nagumo_forward/                   # Case 1 — translation-invariant Nagumo PDE (forward)
│   ├── 01_nagumo_inverse/                   # Case 1 (cont.) — parameter inference D, sparse data
│   │   └── vanilla_PINN_baseline/           # Vanilla-PINN comparison for the inverse problem (Fig. 2)
│   ├── 02_diffusion_2D/                     # Case 2 — 2D linear diffusion, self-similar
│   ├── 03_porous_medium_2D/                 # Case 3 — 2D porous-medium PDE in polar coords
│   ├── 04_burgers/                          # Case 4 — Burgers PDE (translation + scaling)
│   └── 05_gKdV_steady_blowup/               # Case 5 — generalized KdV, steady-state self-similar blow-up
├── supplementary/                           # SI Secs. 1–2
│   ├── SI1_diffusion_1D/                    # 1D diffusion self-similar (SI Sec. 1, Fig. SI-1)
│   └── SI2_porous_medium_axisymmetric/      # Axisymmetric 2D PME, 2nd-kind self-similarity (SI Sec. 2, Fig. SI-2)
├── baselines/                               # Reference numerical methods (SI Sec. 3)
│   ├── nagumo_FD/                           # Backward-Euler / central-FD for Nagumo MN frame (SI Sec. 3.1)
│   ├── porous_medium_axisymmetric_FD/       # FD for axisymmetric PME ("Graveleau" solutions, SI Sec. 3.2)
│   ├── porous_medium_2D_polar_FE/           # COMSOL FE model for the 2D PME polar problem (SI Sec. 3.3)
│   └── burgers_FD/                          # FD with backward Euler for Burgers (SI Sec. 3.4)
└── data/
    └── from_COMSOL/                         # Auxiliary COMSOL reference models cited in the paper
```

Each PINN case-study folder follows the same pattern:

* `main.m`       — defines networks, collocation points, boundary/initial conditions, and runs L-BFGS training.
* `modelLoss.m`  — defines the total loss = PDE residual + algebraic constraint residual + boundary + initial condition residuals.
* `*.mat`        — warm-start / pretraining files used by `main.m` (loaded via `load <name>`).

## How to Reproduce

1. Open MATLAB and `cd` into the case-study directory you wish to reproduce.
2. Run `main` (or `main.m`). The script will load the relevant warm-start `.mat` file (if any), construct the two networks, build collocation grids, and execute L-BFGS for the number of epochs reported in the paper.
3. Trained variables (`parameters`, `net1`, `net2`, `Loss`, …) remain in the workspace and can be used directly to plot:
   * `Loss` — convergence curves;
   * `forward(net1, …)` — the rescaled solution `w(y, tau)`;
   * `forward(net2, …)` — the time-dependent symmetry parameters.

Some scripts contain a `return` statement immediately before the training loop. This was used during interactive development to stop after the setup phase; **delete or comment out that single `return` line** in `main.m` before the first call if you wish to train from scratch (this is needed in `case_studies/02_diffusion_2D/main.m`). All other scripts run end-to-end as-is.

### Mapping from manuscript content to scripts

| Section / Figure                                         | Case-study folder                                            | Notes |
|----------------------------------------------------------|--------------------------------------------------------------|-------|
| _"Case study 1"_ — **Nagumo**, Fig. 1                         | `case_studies/01_nagumo_forward/`                            | Forward problem; trains for 10,000 L-BFGS epochs over `y ∈ [-30, 30]`, `t ∈ [0, 20]`. |
| §"Parameter inference in the co-moving frame", Fig. 2    | `case_studies/01_nagumo_inverse/`                            | Inverse problem (infer `D`); sparse synthetic data are in `data_for_Nagumo_fifth.mat` (loaded automatically). The `vanilla_PINN_baseline/` subfolder reproduces the vanilla PINN comparison. |
| _"Case study 2"_ — **2D Diffusion**, Fig. 3                   | `case_studies/02_diffusion_2D/`                              | Trains a `N_w(xi, eta, tau)` with 3 hidden layers × 40 neurons. Comment out the `return` statement to start training. Loads `result_ti5.mat` as a warm start. |
| _"Case study 3"_ — **2D Porous Medium**, Fig. 4               | `case_studies/03_porous_medium_2D/`                          | Polar-coordinate PME with orthogonality template; loads `result_test.mat`. |
| _"Case study 4"_ — **Burgers**, Fig. 5                        | `case_studies/04_burgers/`                                   | 5,000 L-BFGS epochs; no warm start required. `result_10000_for_paper.mat` is provided for direct post-processing. |
| _"Case study 5"_ — **gKdV steady-state**, Fig. 6              | `case_studies/05_gKdV_steady_blowup/`                        | Steady-state PINN that treats the blow-up rate `G` as a trainable parameter. Several warm-start weight files (`init_weights_*.mat`) are provided, matching the four pretraining strategies in SI §4.2. |
| SI §1 — **1D Diffusion (self-similar)**, Fig. SI-1           | `supplementary/SI1_diffusion_1D/`                            | 15,000 L-BFGS epochs; `result_15000_for_paper.mat` provided. |
| SI §2 — **Axisymmetric 2D Porous Medium**, Fig. SI-2         | `supplementary/SI2_porous_medium_axisymmetric/`              | 15,000 L-BFGS epochs; loads `result_7_new.mat` as warm start; `result_15000_for_paper.mat` provided. |
| SI §3.1 — **Nagumo FD baseline**, Fig. SI-3                  | `baselines/nagumo_FD/`                                        | Implicit backward-Euler / central-FD reference. Run `main_nagumo.m`. |
| SI §3.2 — **Axisymmetric PME ("Graveleau") FD**, Fig. SI-4   | `baselines/porous_medium_axisymmetric_FD/`                    | FD-MN reference for the rescaled axisymmetric PME. Run `main.m`. |
| SI §3.3 — **2D PME polar FE baseline**, Fig. SI-5            | `baselines/porous_medium_2D_polar_FE/`                        | COMSOL Multiphysics `.mph` files (open in COMSOL ≥ 6.0). |
| SI §3.4 — **Burgers FD baseline**, Fig. SI-6                 | `baselines/burgers_FD/`                                       | Run `main.m`. |
| SI §4.1 — **Robustness to random initialization (Burgers)**  | `case_studies/04_burgers/`                                   | Repeat `main.m` with different random seeds (`rng(seed); ...`). |
| SI §4.2 — **KdV warm-up sensitivity**, Fig. SI-7             | `case_studies/05_gKdV_steady_blowup/`                        | Switch the `load init_weights_*` line in `main.m` between `init_weights_TW.mat`, `init_weights_gaussian_dev_05.mat`, `init_weights_gaussian_dev_2.mat`, and `initial_weights_gaussian.mat`. |

### Plotting the figures

The published figures were prepared interactively in MATLAB from the workspace variables after running each `main.m`. Typical plotting commands:

```matlab
% Example — Burgers (Fig. 5)
[X, T] = ndgrid(linspace(-6, 6, 601), linspace(0, 6, 301));
W = extractdata(forward(net1, dlarray([T(:)'; X(:)'], "CB")));
W = reshape(W, size(X));
figure; pcolor(T, X, W); shading flat; colorbar;
xlabel('\tau'); ylabel('y'); title('w(y,\tau)');
```

Each case-study folder also contains the `.mat` files needed to re-load the trained networks and reproduce the exact figures without re-training. The `result_*_for_paper.mat` files (where included) correspond to the runs used to generate the published figures.

## Notes, Assumptions, and Limitations

* **MATLAB versions.** The code uses `lbfgsupdate` and `lbfgsState`, which were introduced in MATLAB R2023a. Earlier versions are not supported.
* **Reproducibility of weights.** L-BFGS is deterministic given identical initial weights and collocation points. The provided `.mat` files freeze those initial states so that retraining converges to the figures shown. Random-initialization studies (SI §4.1) deliberately vary the seed.
* **Warm-start workflow.** Several `main.m` scripts continue from a previously saved checkpoint via a `load <name>` statement. The required `.mat` files are bundled in this repository alongside the relevant `main.m`.
* **`return` statements.** A few scripts (notably `02_diffusion_2D/main.m`) contain a `return` line before the training loop, leftover from interactive use. Comment it out before training. This is the only manual edit required.
* **Computational cost.** Training times are CPU-bound and range from ~15 s (Nagumo forward, simple cases) to ~150 min (PME polar, gKdV blow-up). They match the wall-clock numbers reported in the manuscript.
* **Symbolic exponents.** Self-similar exponents are inferred from the steady-state ratio `G_ss / C_ss` of the auxiliary network `N_p(τ)`. Post-processing is described inline in each case-study section of the manuscript / SI.
* **COMSOL files** require the user to have a COMSOL Multiphysics license. They are provided strictly to document the FE baselines reported in SI §3.3; the PINN reproducibility itself does not depend on them.

## Citation

If you use this code, please cite the manuscript above (citation details to be updated upon publication).

---

## MIT License

Copyright (c) 2026 The authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---

## Disclaimer

This software is provided "as is" without any express or implied warranties.
This includes, but is not limited to, warranties of merchantability, fitness for a particular purpose, and non-infringement.
The authors and copyright holders are not liable for any claims, damages, or other liabilities arising from the use of this software

Last revised by M. Kavousanakis, July 2, 2026

## Contact

For questions concerning reproduction of any result, contact the corresponding author (`yannisk@jhu.edu`).
