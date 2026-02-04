"""
Stage I 的 Objective（目标函数）实现：阈值过滤 + 排序/评分。

对应规格 `工具描述文档.md` 的要求：
1) Constraints（阈值过滤）：不满足直接淘汰（infeasible）
2) Score 或 Lexicographic 排序：在可行集合内选 best/top-k

这里默认实现的是 lexicographic（多指标字典序），因为：
- 更容易跑通闭环，不用纠结权重
- 也更符合“先满足可行性，再比速度/质量”的论文叙述

如需 Optuna/TPE 等贝叶斯调参，可切换为 score（标量）或在此基础上扩展。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class RankedRow:
    """内部结构：记录一条 candidate 记录是否可行以及其排序 key。"""
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
    对 candidate 聚合指标表做筛选与排序，返回 top-k 以及 best（第一名）。

    参数:
      candidate_df: candidate 级别的 DataFrame（通常来自 collect.aggregate_mean_std）
      objective_cfg: Hydra 配置里的 objective（mode/constraints/key 等）
      top_k: 需要保留的 top-k 数量
      id_field: candidate 的 ID 字段名（默认 candidate_id）

    返回:
      - topk_df: 从 candidate_df 里筛出的 top-k 子集（附加 _feasible/_rank_key 调试列）
      - best_row: best 的原始 record dict（None 表示没有候选）
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
        # 先做阈值过滤（可行性），再计算排序 key。
        feasible = _check_constraints(r, constraints)
        ranked.append(RankedRow(row=r, feasible=feasible, key=ranker(r)))

    feasible_ranked = [x for x in ranked if x.feasible]
    # 如果存在可行解：只在可行集合里排序；否则对全部候选排序（便于调试/避免空输出）。
    ordered = sorted(feasible_ranked if feasible_ranked else ranked, key=lambda x: x.key)

    top = ordered[: max(1, int(top_k))]
    top_ids = {t.row.get(id_field) for t in top}

    out_df = candidate_df[candidate_df[id_field].isin(top_ids)].copy()
    # 下面两个列是为了让 topk.yaml/CSV 更可解释：为什么它是 top-k。
    out_df["_feasible"] = out_df[id_field].apply(lambda cid: any(x.row.get(id_field) == cid and x.feasible for x in top))
    out_df["_rank_key"] = out_df[id_field].apply(lambda cid: next((x.key for x in top if x.row.get(id_field) == cid), ()))

    best = top[0].row if top else None
    return out_df, best


def _check_constraints(row: dict[str, Any], constraints: list[Any]) -> bool:
    """检查一条 candidate 记录是否满足所有阈值约束（缺值/NaN 默认不通过）。"""
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
    把 NaN/None 映射到可排序的数值，避免排序时报错。

    nan_policy:
      - "low": NaN 视为极小（更差/更不优先，具体取决于你怎么构造 key）
      - "high": NaN 视为极大
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
    """构造一个“字典序排序 key”函数。"""
    def _key(row: dict[str, Any]) -> tuple[Any, ...]:
        out: list[Any] = []
        for spec in key_spec:
            field = str(getattr(spec, "field", ""))
            order = str(getattr(spec, "order", "asc")).lower()
            nan_policy = str(getattr(spec, "nan", "high")).lower()
            raw = row.get(field, float("nan"))
            val = _nan_order(raw, nan_policy=nan_policy, order=order)
            # 统一转换为“越小越好”的排序 key：
            # - 如果该字段要 desc：取负号
            # - 如果要 asc：保持原值
            out.append(-val if order == "desc" else val)
        return tuple(out)

    return _key


def _score_ranker(score_cfg: Any) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    """构造一个标量 score 的排序 key（更适合 Optuna/TPE）。"""
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
