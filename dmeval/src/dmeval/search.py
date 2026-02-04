"""
Stage I（tune）里的候选超参生成（SearchStrategy）。

设计目标：
- 提供一个最小但可扩展的候选生成器：grid / random（optuna 预留）
- 生成的单位是 Candidate：
  - candidate_id: 稳定的字符串 ID（便于落盘、复现）
  - patch: 一个 YAML patch dict（会合并到 base_cfg 生成 cfg_inference.yaml）
  - params_flat: 扁平的参数字典（便于记录/打印/调试）

注意：
- 这里不做任何“评价/排序”，只负责生成候选。
- 评价单位在 L1 里是 candidate（在 seeds 上完整跑完后再聚合/排名）。
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

from .util import merge_patches, nested_dict_from_dotpath


@dataclass(frozen=True)
class Candidate:
    """一个候选配置（candidate）。"""
    candidate_id: str
    patch: dict[str, Any]
    params_flat: dict[str, Any]


def _as_list(value: Any) -> list[Any]:
    """把标量转成单元素 list，方便统一处理 grid space。"""
    if isinstance(value, list):
        return value
    return [value]


def generate_candidates(*, base_patch: dict[str, Any], search_cfg: Any) -> list[Candidate]:
    """
    根据 search_cfg 生成候选列表。

    - grid: 传入离散列表，做笛卡尔积
    - random: 支持离散 choices 或简单的 float/int 分布采样
    - optuna: 预留接口（当前最小实现未启用）
    """
    strategy = str(getattr(search_cfg, "strategy", "grid")).lower()
    if strategy == "grid":
        space = getattr(search_cfg, "space", {})
        return list(_grid_candidates(base_patch=base_patch, space=space))
    if strategy == "random":
        space = getattr(search_cfg, "space", {})
        n_candidates = int(getattr(search_cfg, "n_candidates", 10))
        seed = getattr(search_cfg, "seed", None)
        return list(_random_candidates(base_patch=base_patch, space=space, n=n_candidates, seed=seed))
    if strategy == "optuna":
        raise NotImplementedError(
            "optuna strategy is not implemented in this minimal build; install dmeval[optuna] and extend search.py"
        )
    raise ValueError(f"Unknown search strategy: {strategy}")


def _grid_candidates(*, base_patch: dict[str, Any], space: Any) -> Iterable[Candidate]:
    """grid 搜索：对每个参数维度取离散列表，做笛卡尔积。"""
    if not isinstance(space, dict) or not space:
        yield Candidate(candidate_id="c0000", patch=base_patch, params_flat={})
        return

    keys = list(space.keys())
    values_lists = [_as_list(space[k]) for k in keys]

    for i, combo in enumerate(itertools.product(*values_lists)):
        params = dict(zip(keys, combo))
        patch_parts = [base_patch]
        for k, v in params.items():
            # 把 "a.b.c" 这种 dotpath 参数变成嵌套 dict patch，便于合并到 YAML。
            patch_parts.append(nested_dict_from_dotpath(str(k), v))
        patch = merge_patches(*patch_parts)
        yield Candidate(candidate_id=f"c{i:04d}", patch=patch, params_flat=params)


def _random_candidates(*, base_patch: dict[str, Any], space: Any, n: int, seed: Any) -> Iterable[Candidate]:
    """random 搜索：按给定分布/choices 采样 n 个候选。"""
    if seed is not None:
        random.seed(int(seed))

    if not isinstance(space, dict) or not space:
        for i in range(n):
            yield Candidate(candidate_id=f"c{i:04d}", patch=base_patch, params_flat={})
        return

    for i in range(n):
        params: dict[str, Any] = {}
        for key, spec in space.items():
            params[str(key)] = _sample_one(spec)
        patch_parts = [base_patch]
        for k, v in params.items():
            patch_parts.append(nested_dict_from_dotpath(k, v))
        patch = merge_patches(*patch_parts)
        yield Candidate(candidate_id=f"c{i:04d}", patch=patch, params_flat=params)


def _sample_one(spec: Any) -> Any:
    """
    从一个参数 spec 采样一个值。

    支持的 spec 形式：
    - list: 从列表里随机选
    - dict:
      - {"choices": [...]}：离散选项
      - {"type": "int", "low": 1, "high": 10}
      - {"type": "float", "low": 1e-3, "high": 1e-1, "log": true}
    """
    if isinstance(spec, list):
        return random.choice(spec)
    if isinstance(spec, dict):
        kind = str(spec.get("type", "")).lower()
        if "choices" in spec:
            return random.choice(list(spec["choices"]))
        if kind in ("int", "integer"):
            low = int(spec["low"])
            high = int(spec["high"])
            return random.randint(low, high)
        if kind in ("float", "number"):
            low = float(spec["low"])
            high = float(spec["high"])
            if bool(spec.get("log", False)):
                if low <= 0 or high <= 0:
                    raise ValueError("log-uniform requires low/high > 0")
                return math.exp(random.uniform(math.log(low), math.log(high)))
            return random.uniform(low, high)
    return spec
