"""
指标采集后的 DataFrame 处理：trial → run/candidate 聚合、CSV 写出。

重要原则（对应 `工具描述文档.md`）：
- DMEval **不重算指标**：这里只做聚合统计（mean/std）与格式化输出
- 分两层 CSV：
  - trial_metrics.csv：每个 trial 一行（来自 adapter 抽取）
  - run_metrics.csv：按 (scenario, run_tag, seed, ...) 聚合（公平比较/排名/绘图用）

Stage I 额外输出 candidate_metrics.csv：
- 按 candidate 聚合（跨 seed），用于 objective（阈值 + 排序）选择 best/top-k
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .util import ensure_dir


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """把 adapter 输出的 list[dict] 转为 DataFrame。"""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(rows)
    return df


def coerce_numeric(df: pd.DataFrame, *, exclude: list[str]) -> pd.DataFrame:
    """
    尝试把非排除列转换成数值型（无法转换则保留原样）。

    注：当前实现里我们尽量依赖 adapter 输出的类型；
    这个函数更多是容错/兼容用（例如 jsonl 里数字可能是字符串）。
    """
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def aggregate_mean_std(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    """
    对数值列做 mean/std 聚合，并把多级列名拍平成 `<col>_mean` / `<col>_std`。

    参数:
      group_cols: 聚合维度（例如 run 级别可用 ["scenario","sampler","run_tag","seed"]）
    """
    if df.empty:
        return df
    df2 = df.copy()
    # 只对数值列做聚合；文本/类别列只用于 group key。
    numeric_cols = [c for c in df2.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df2[c])]
    agg = df2.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([x for x in col if x]) if isinstance(col, tuple) else str(col)  # type: ignore[arg-type]
        for col in agg.columns
    ]
    return agg


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """写 CSV（不包含 index）。"""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_stage_metrics(
    *,
    out_dir: Path,
    trial_df: pd.DataFrame,
    run_df: pd.DataFrame,
    extra: dict[str, pd.DataFrame] | None = None,
) -> None:
    """按约定文件名写出一个 stage 的核心产物（trial/run + 额外表）。"""
    ensure_dir(out_dir)
    write_csv(trial_df, out_dir / "trial_metrics.csv")
    write_csv(run_df, out_dir / "run_metrics.csv")
    if extra:
        for name, df in extra.items():
            write_csv(df, out_dir / f"{name}.csv")
