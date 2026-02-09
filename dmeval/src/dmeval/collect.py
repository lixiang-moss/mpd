"""
Post-collection DataFrame processing: trial → run/candidate aggregation and CSV writing.

Key principles:
- DMEval **does not recompute metrics**: we only aggregate statistics (mean/std) and format outputs
- Two CSV levels:
  - `trial_metrics.csv`: one row per trial (from adapter extraction)
  - `run_metrics.csv`: aggregated by (scenario, run_tag, seed, ...) for fair comparison/ranking/plotting

Stage I additionally outputs `candidate_metrics.csv`:
- aggregated by candidate (across seeds), used by the objective (constraints + ranking) to select best/top-k
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .util import ensure_dir


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert adapter output (list[dict]) into a DataFrame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(rows)
    return df


def coerce_numeric(df: pd.DataFrame, *, exclude: list[str]) -> pd.DataFrame:
    """
    Try to coerce non-excluded columns to numeric (keep original values on failure).

    Note: we try to rely on adapter-provided types. This is mostly for robustness/compatibility
    (e.g., numbers may arrive as strings from JSONL).
    """
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def aggregate_mean_std(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate numeric columns with mean/std and flatten multi-index columns to `<col>_mean` / `<col>_std`.

    Args:
      group_cols: grouping dimensions (e.g. run-level: ["scenario","sampler","run_tag","seed"])
    """
    if df.empty:
        return df
    df2 = df.copy()
    # Aggregate only numeric columns; text/categorical columns are used as group keys only.
    numeric_cols = [c for c in df2.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df2[c])]
    agg = df2.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([x for x in col if x]) if isinstance(col, tuple) else str(col)  # type: ignore[arg-type]
        for col in agg.columns
    ]
    return agg


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV (without index)."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_stage_metrics(
    *,
    out_dir: Path,
    trial_df: pd.DataFrame,
    run_df: pd.DataFrame,
    extra: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Write the core artifacts for a stage using the agreed filenames (trial/run + extra tables)."""
    ensure_dir(out_dir)
    write_csv(trial_df, out_dir / "trial_metrics.csv")
    write_csv(run_df, out_dir / "run_metrics.csv")
    if extra:
        for name, df in extra.items():
            write_csv(df, out_dir / f"{name}.csv")
