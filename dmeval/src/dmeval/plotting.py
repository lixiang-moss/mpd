"""
Plotting utilities (minimal implementation).

Goals:
- Produce the minimal plots required by the spec:
  - bar plots for success / fraction_valid / time
  - scatter plot for time vs success

Notes:
- Kept intentionally simple to avoid heavy plotting dependencies (L1: make the loop work first)
- Paper-quality styling/layout (seaborn/plotly/LaTeX, etc.) can be added later
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .util import ensure_dir


def plot_stage_compare(*, run_metrics_agg: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot Stage-II artifacts.

    Args:
      run_metrics_agg: DataFrame aggregated by (scenario, sampler) (typically from `collect.aggregate_mean_std`)
      out_dir: output directory (writes multiple PNGs)
    """
    if run_metrics_agg.empty:
        return
    ensure_dir(out_dir)

    required_cols = {"scenario", "sampler"}
    if not required_cols.issubset(set(run_metrics_agg.columns)):
        return

    scenarios = sorted(run_metrics_agg["scenario"].dropna().unique().tolist())
    for scenario in scenarios:
        sdf = run_metrics_agg[run_metrics_agg["scenario"] == scenario].copy()
        if sdf.empty:
            continue
        # Use a stable filename scheme `<scenario>__<metric>.png` for easy scripting/paper references.
        _barplot(sdf, x="sampler", y="success_mean", title=f"{scenario}: success", out=out_dir / f"{scenario}__success.png")
        _barplot(
            sdf,
            x="sampler",
            y="fraction_valid_mean",
            title=f"{scenario}: fraction_valid",
            out=out_dir / f"{scenario}__fraction_valid.png",
        )
        _barplot(
            sdf,
            x="sampler",
            y="t_inference_total_mean",
            title=f"{scenario}: inference time",
            out=out_dir / f"{scenario}__time.png",
        )
        _scatter(
            sdf,
            x="t_inference_total_mean",
            y="success_mean",
            label_col="sampler",
            title=f"{scenario}: time vs success",
            out=out_dir / f"{scenario}__time_vs_success.png",
        )


def _barplot(df: pd.DataFrame, *, x: str, y: str, title: str, out: Path) -> None:
    """Simple bar-plot helper (one bar per sampler)."""
    if y not in df.columns:
        return
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.bar(df[x].astype(str), df[y])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _scatter(df: pd.DataFrame, *, x: str, y: str, label_col: str, title: str, out: Path) -> None:
    """Scatter plot helper (annotates each point with the sampler name)."""
    if x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    plt.title(title)
    for _, r in df.iterrows():
        plt.scatter(r[x], r[y])
        plt.text(r[x], r[y], str(r.get(label_col, "")), fontsize=8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
