#!/usr/bin/env python3
"""
Dummy Planner（仅用于自检/单测）。

用途：
- 在不安装 MPD / 不依赖 torch / 不需要 GPU 的情况下，快速验证 DMEval 的闭环是否正确：
  Stage I tune -> best_configs -> Stage II compare -> CSV/plots/manifest

行为：
- 接收与 MPD inference.py 类似的一组 CLI 参数（cfg_inference_path/results_dir/seed/N/device...）
- 在 `<results_dir>/<seed>/` 下写出：
  - args_inference.yaml
  - trial_metrics.jsonl（每个 trial 一行）

注意：
- 该脚本不生成 `.pt`，因此 DMEval 会走 Adapter 的 jsonl 分支（更轻量）
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import yaml


def _load_cfg(path: Path) -> dict[str, Any]:
    """读取 cfg_inference.yaml（这里只需要少数字段来模拟“速度-质量”关系）。"""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: Any) -> None:
    """写 YAML（保持 key 顺序，允许中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 jsonl（每行一个 trial 指标 dict）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    """dummy planner 主入口（保持与 MPD 相同/相近的参数形状，便于被 runner.py 调用）。"""
    parser = argparse.ArgumentParser(description="Dummy planner that mimics MPD outputs (jsonl only).")
    parser.add_argument("--cfg_inference_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection_start_goal", default="validation")
    parser.add_argument("--n_start_goal_states", type=int, default=1)
    parser.add_argument("--save_results_single_plan_low_mem", default="true")
    parser.add_argument("--device", default="cpu")
    args, _unknown = parser.parse_known_args()

    cfg = _load_cfg(Path(args.cfg_inference_path))
    diffusion_sampling_method = str(cfg.get("diffusion_sampling_method", "ddim"))
    planner_alg = str(cfg.get("planner_alg", "mpd"))

    # 构造一个简单的“速度-质量”玩具模型：
    # - steps 越多，time 越大
    # - steps 越多，success 概率越大但会饱和
    steps = None
    if diffusion_sampling_method == "ddim":
        steps = int(((cfg.get("ddim") or {}).get("ddim_sampling_timesteps") or 10))
    elif diffusion_sampling_method in ("dpm_solver", "dpm_solver_pp"):
        steps = int(((cfg.get("dpm_solver") or {}).get("dpm_solver_steps") or 20))
    else:
        steps = 10

    # 通过 seed + sampler 做确定性随机，保证同一配置可复现。
    random.seed(int(args.seed) + 1000 * int(abs(hash(diffusion_sampling_method)) % 1000))

    # 生成 toy 指标分布。
    base_time = 0.05 * steps
    base_success = 1.0 - math.exp(-steps / 15.0)
    base_success = max(0.0, min(1.0, base_success))

    seed_dir = Path(args.results_dir) / str(args.seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(
        seed_dir / "args_inference.yaml",
        {
            "planner_alg": planner_alg,
            "diffusion_sampling_method": diffusion_sampling_method,
            "selection_start_goal": args.selection_start_goal,
            "n_start_goal_states": int(args.n_start_goal_states),
            "device": args.device,
        },
    )

    # 生成 N 行 trial 指标（字段名与 DMEval 期望的口径一致）。
    rows: list[dict[str, Any]] = []
    for trial_id in range(int(args.n_start_goal_states)):
        jitter = random.random() * 0.02
        t_total = base_time * (1.0 + jitter)
        success = 1.0 if random.random() < base_success else 0.0
        fraction_valid = max(0.0, min(1.0, success * (0.7 + 0.3 * random.random())))
        collision_intensity = max(0.0, (1.0 - fraction_valid) * random.random() * 0.2)

        rows.append(
            {
                "trial_id": trial_id,
                "t_inference_total": t_total,
                "t_generator": t_total * 0.7,
                "t_guide": t_total * 0.3,
                "success": success,
                "fraction_valid": fraction_valid,
                "collision_intensity": collision_intensity,
                "ee_pose_goal_error_position_norm_best": (1.0 - success) * (0.1 + random.random()),
                "ee_pose_goal_error_orientation_norm_best": (1.0 - success) * (0.1 + random.random()),
                "path_length_best": 1.0 + random.random(),
                "smoothness_best": 1.0 + random.random(),
                "diversity": random.random(),
            }
        )

    # 模拟一点点运行耗时，避免某些环境里“0 秒完成”导致图表/排序不明显。
    time.sleep(0.01)
    _write_jsonl(seed_dir / "trial_metrics.jsonl", rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
