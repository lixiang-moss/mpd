"""
可视化模块（最小实现）。

目标：
- 输出 `工具描述文档.md` 要求的基础图：
  - success / fraction_valid / time 的柱状图
  - time vs success 的散点图

注意：
- 这里刻意保持简单，避免引入更重的可视化框架（L1 先跑通闭环）
- 论文级别的图表美化/排版（如 seaborn/plotly/latex）可以后续增强
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .util import ensure_dir


def plot_stage_compare(*, run_metrics_agg: pd.DataFrame, out_dir: Path) -> None:
    """
    为 Stage II 产物绘图。

    参数:
      run_metrics_agg: 按 (scenario,sampler) 聚合后的 DataFrame（通常来自 collect.aggregate_mean_std）
      out_dir: 输出目录（会写入多张 png）
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
        # 统一输出文件名 `<scenario>__<metric>.png`，方便脚本/论文引用。
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
    """最简单的柱状图封装（每个 sampler 一根柱）。"""
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
    """散点图（并在点旁边标注 sampler 名称）。"""
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
