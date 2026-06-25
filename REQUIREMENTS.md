# Requirements

This is a MATLAB project. There is no `requirements.txt`; the dependencies are MATLAB toolboxes.

## MATLAB

* **MATLAB R2023a or later** (R2024a tested). Earlier versions do not provide `lbfgsupdate` / `lbfgsState`.

## Required toolboxes

| Toolbox                                  | Used for                                                              |
|------------------------------------------|-----------------------------------------------------------------------|
| Deep Learning Toolbox                    | `dlnetwork`, `dlarray`, `dlfeval`, `lbfgsupdate`, `featureInputLayer`, `fullyConnectedLayer`, `tanhLayer`, `dlaccelerate`, `dlgradient` |
| Statistics and Machine Learning Toolbox  | Auxiliary numerical utilities                                         |

## Optional

| Tool                                     | Used for                                                              |
|------------------------------------------|-----------------------------------------------------------------------|
| COMSOL Multiphysics ≥ 6.0                | Opening `.mph` baseline files in `baselines/porous_medium_2D_polar_FE/` and `data/from_COMSOL/` |

## How to verify your installation

In MATLAB, run:

```matlab
ver               % lists available toolboxes
which lbfgsupdate % must return a path under Deep Learning Toolbox
```

If `lbfgsupdate` is not found, upgrade MATLAB to R2023a or later, or install the Deep Learning Toolbox via the Add-On Explorer.
