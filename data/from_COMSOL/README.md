# Auxiliary COMSOL reference models

The COMSOL Multiphysics (`.mph`) reference files used during the development
of the manuscript are bundled here for completeness:

| File                                          | Used for                                              |
|-----------------------------------------------|-------------------------------------------------------|
| `nagumo_works.mph`                            | Manuscript Case 1 FE reference solution               |
| `diffusion1d_works.mph`                       | SI §1 FE reference                                    |
| `diffusion2d_works.mph`                       | Manuscript Case 2 FE reference                        |
| `burgers_fails_for_small_nu.mph`              | Manuscript Case 4 / SI §3.4 baseline                  |

Open the files in COMSOL Multiphysics ≥ 6.0 (a COMSOL license is required).

None of these are strictly required to reproduce the PINN results — they only
underpin the reference / baseline figures discussed in the SI. The MATLAB
finite-difference baselines that *are* required to reproduce SI Fig. SI-3
through SI-6 are bundled directly in `../../baselines/`.
