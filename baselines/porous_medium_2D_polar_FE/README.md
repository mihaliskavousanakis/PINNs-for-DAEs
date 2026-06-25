# 2D porous-medium polar — finite-element baseline (SI §3.3)

This folder bundles the COMSOL Multiphysics model referenced in SI §3.3:

* `porous_medium2d_failure.mph`

Open in COMSOL Multiphysics ≥ 6.0. It implements the FE
discretisation described in SI §3.3 and demonstrate the convergence failure
of the conventional solver beyond `τ ≈ 0.68`, which is the qualitative
result shown in Fig. SI-5.

If you wish to rebuild the FE baseline from scratch, follow SI §3.3:

* PDE: Eq. (18) of the main manuscript (2D PME in polar coordinates).
* Domain: `(ρ, θ) ∈ [0, 4] × [0, π/2]`.
* Discretisation: 501 elements in `ρ`, 51 elements in `θ`, quadratic basis
  functions (~10⁵ collocation points).
* Time integration: fully implicit backward Euler, `Δt = 10⁻³`.
* Integral constraint (orthogonality, Eq. 19) approximated with the
  trapezoidal rule.
* Boundary conditions: see SI §3.3.

Total simulation cost reported in SI §3.3 is ≈ 15 minutes wall-clock.

A COMSOL Multiphysics license is required to open the `.mph` file. The
PINN reproducibility itself does not depend on these models — they only
underpin the comparison plotted in Fig. SI-5.
