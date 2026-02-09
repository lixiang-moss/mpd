#!/usr/bin/env python3
"""
Dummy planner (for smoke tests / unit tests only).

Use case:
- Quickly validate the DMEval end-to-end loop without installing MPD, without torch, and without a GPU:
  Stage I tune -> best_configs -> Stage II compare -> CSV/plots/manifest

Behavior:
- Accepts a CLI shape similar to MPD's `inference.py` (cfg_inference_path/results_dir/seed/N/device...)
- Writes the following under `<results_dir>/<seed>/`:
  - args_inference.yaml
  - trial_metrics.jsonl (one JSON object per trial)

Note:
- This script does not generate `.pt` files, so DMEval will exercise the adapter's JSONL path (lighter-weight).
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
    """Load `cfg_inference.yaml` (only a few fields are needed for the toy speed/quality model)."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: Any) -> None:
    """Write YAML (preserve key order; allow Unicode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL (one trial-metrics dict per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    """Dummy planner entrypoint (keeps a CLI similar to MPD so it can be called by `runner.py`)."""
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

    # A simple toy "speed vs. quality" model:
    # - more steps => more time
    # - more steps => higher success probability (with saturation)
    steps = None
    if diffusion_sampling_method == "ddim":
        steps = int(((cfg.get("ddim") or {}).get("ddim_sampling_timesteps") or 10))
    elif diffusion_sampling_method in ("dpm_solver", "dpm_solver_pp"):
        steps = int(((cfg.get("dpm_solver") or {}).get("dpm_solver_steps") or 20))
    else:
        steps = 10

    # Deterministic RNG based on (seed, sampler) for reproducibility.
    random.seed(int(args.seed) + 1000 * int(abs(hash(diffusion_sampling_method)) % 1000))

    # Generate a toy metrics distribution.
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

    # Generate N trial rows (field names match what DMEval expects).
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

    # Add a tiny delay so plots/rankings are not trivially identical in some environments.
    time.sleep(0.01)
    _write_jsonl(seed_dir / "trial_metrics.jsonl", rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
