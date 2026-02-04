from __future__ import annotations

from typing import Any, Dict

from dmeval.utils import is_nan, to_float


def is_feasible(run_row: Dict[str, Any]) -> bool:
    """
    A minimal feasibility filter for Stage-I tuning.

    You will likely replace this with your real constraints, e.g.:
    - success_mean >= threshold
    - fraction_valid_mean >= threshold
    - collision_intensity_mean <= threshold
    """
    success = to_float(run_row.get("success_mean", float("nan")))
    if is_nan(success):
        return False
    return success > 0.0


def score(run_row: Dict[str, Any]) -> float:
    """
    A simple single-objective score (higher is better).

    Default heuristic:
    - prioritize success (0..1)
    - lightly penalize inference time (seconds)

    Replace this with your real objective later.
    """
    success = to_float(run_row.get("success_mean", float("nan")))
    time_sec = to_float(run_row.get("t_inference_total_mean", float("nan")))
    if is_nan(success):
        return float("-inf")
    if is_nan(time_sec):
        time_sec = 0.0

    return float(success - 1e-3 * time_sec) #临时占位

