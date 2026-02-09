"""
DMEval vNext (L1) package entrypoint.

This package implements the DMEval pipeline described in the project spec:
- Stage I: tune (hyperparameter search; outputs `best_configs/`)
- Stage II: compare (loads `best_configs/` and runs fair comparisons across scenarios)

Important: DMEval **does not compute metrics**. It only:
1) Runs the system-under-test (e.g., MPD) serially as subprocesses;
2) Extracts metrics from the results directory (via an Adapter);
3) Aggregates/ranks/plots and writes standardized artifacts (CSV/plots/manifests).
"""

__all__ = ["__version__"]

# Version string used in manifests and reproducibility records.
__version__ = "0.1.0"
