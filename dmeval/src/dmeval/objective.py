"""
Stage-I objective implementation: constraints filtering + ranking/scoring.

As required by the spec:
1) Constraints filtering: candidates that violate thresholds become infeasible
2) Score or lexicographic ranking: select best/top-k among feasible candidates

Default mode is lexicographic (multi-metric lexicographic ordering) because:
- It is easier to get an end-to-end loop working without tuning weights
- It matches the typical paper narrative: feasibility first, then speed/quality

For Optuna/TPE (Bayesian) tuning, you can switch to a scalar score objective or extend this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class RankedRow:
    """Internal record: feasibility flag and ranking key for one candidate row."""
    row: dict[str, Any]
    feasible: bool
    key: tuple[Any, ...]


def select_topk(
    *,
    candidate_df: pd.DataFrame,
    objective_cfg: Any,
    top_k: int,
    id_field: str = "candidate_id",
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """
    Filter and rank the candidate-level metrics table and return top-k and the best (rank 1).

    Args:
      candidate_df: candidate-level DataFrame (typically from `collect.aggregate_mean_std`)
      objective_cfg: Hydra objective config (mode/constraints/keys)
      top_k: number of candidates to keep
      id_field: ID field name (default: `candidate_id`)

    Returns:
      - topk_df: top-k subset of `candidate_df` with debugging columns (`_feasible`, `_rank_key`)
      - best_row: the best record dict (None means no candidates)
    """
    if candidate_df.empty:
        return candidate_df, None

    mode = str(getattr(objective_cfg, "mode", "lexicographic")).lower()
    constraints = list(getattr(objective_cfg, "constraints", []) or [])

    records = candidate_df.to_dict(orient="records")

    if mode == "lexicographic":
        key_spec = list(getattr(objective_cfg, "lexicographic_key", []) or [])
        ranker = _lexicographic_ranker(key_spec)
    elif mode == "score":
        score_cfg = getattr(objective_cfg, "score", None)
        ranker = _score_ranker(score_cfg)
    else:
        raise ValueError(f"Unknown objective mode: {mode}")

    ranked: list[RankedRow] = []
    for r in records:
        # Apply constraint feasibility first, then compute ranking key.
        feasible = _check_constraints(r, constraints)
        ranked.append(RankedRow(row=r, feasible=feasible, key=ranker(r)))

    feasible_ranked = [x for x in ranked if x.feasible]
    # If feasible solutions exist: rank only within feasible set; otherwise rank all (debug-friendly).
    ordered = sorted(feasible_ranked if feasible_ranked else ranked, key=lambda x: x.key)

    top = ordered[: max(1, int(top_k))]
    top_ids = {t.row.get(id_field) for t in top}

    out_df = candidate_df[candidate_df[id_field].isin(top_ids)].copy()
    # Add debug columns to make topk.yaml/CSV more explainable.
    out_df["_feasible"] = out_df[id_field].apply(lambda cid: any(x.row.get(id_field) == cid and x.feasible for x in top))
    out_df["_rank_key"] = out_df[id_field].apply(lambda cid: next((x.key for x in top if x.row.get(id_field) == cid), ()))

    best = top[0].row if top else None
    return out_df, best


def _check_constraints(row: dict[str, Any], constraints: list[Any]) -> bool:
    """Check whether a candidate record satisfies all threshold constraints (missing/NaN fails)."""
    for c in constraints:
        field = str(getattr(c, "field", ""))
        op = str(getattr(c, "op", "ge")).lower()
        target = getattr(c, "value", None)
        if not field:
            continue

        val = row.get(field, None)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return False

        try:
            v = float(val)
            t = float(target)
        except Exception:
            return False

        if op in ("ge", ">="):
            ok = v >= t
        elif op in ("gt", ">"):
            ok = v > t
        elif op in ("le", "<="):
            ok = v <= t
        elif op in ("lt", "<"):
            ok = v < t
        elif op in ("eq", "=="):
            ok = v == t
        else:
            raise ValueError(f"Unsupported constraint op: {op}")

        if not ok:
            return False
    return True


def _nan_order(value: Any, *, nan_policy: str, order: str) -> float:
    """
    Map NaN/None to a sortable numeric value to avoid ordering errors.

    nan_policy:
      - "low": treat NaN as extremely small
      - "high": treat NaN as extremely large
    """
    if value is None:
        is_nan = True
    else:
        try:
            is_nan = isinstance(value, float) and math.isnan(value)
        except Exception:
            is_nan = False
    if not is_nan:
        return float(value)

    nan_policy = nan_policy.lower()
    order = order.lower()
    if nan_policy == "low":
        return -math.inf if order == "desc" else -math.inf
    if nan_policy == "high":
        return math.inf if order == "asc" else math.inf
    return math.inf


def _lexicographic_ranker(key_spec: list[Any]) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    """Build a lexicographic ranking-key function."""
    def _key(row: dict[str, Any]) -> tuple[Any, ...]:
        out: list[Any] = []
        for spec in key_spec:
            field = str(getattr(spec, "field", ""))
            order = str(getattr(spec, "order", "asc")).lower()
            nan_policy = str(getattr(spec, "nan", "high")).lower()
            raw = row.get(field, float("nan"))
            val = _nan_order(raw, nan_policy=nan_policy, order=order)
            # Normalize to a "smaller is better" key:
            # - for desc fields: negate
            # - for asc fields: keep as-is
            out.append(-val if order == "desc" else val)
        return tuple(out)

    return _key


def _score_ranker(score_cfg: Any) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    """Build a scalar-score ranking key (more suitable for Optuna/TPE)."""
    weights = dict(getattr(score_cfg, "weights", {}) or {})
    maximize = bool(getattr(score_cfg, "maximize", True))

    def _key(row: dict[str, Any]) -> tuple[Any, ...]:
        score = 0.0
        for field, w in weights.items():
            val = row.get(field, float("nan"))
            try:
                v = float(val)
            except Exception:
                v = float("nan")
            if math.isnan(v):
                v = 0.0
            score += float(w) * v
        return (-score,) if maximize else (score,)

    return _key
