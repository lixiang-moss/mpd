import argparse
import csv
import math
import os
from typing import Any, Dict, List, Tuple

import torch
import yaml


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def _is_nan(value: float) -> bool:
    return math.isnan(value)


def _load_args_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _find_results_files(results_root: str) -> List[Tuple[str, str]]:
    hits = []
    for root, _, files in os.walk(results_root):
        for fname in files:
            if fname.startswith("results_single_plan-") and fname.endswith(".pt"):
                hits.append((root, os.path.join(root, fname)))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True, help="Root directory of sweep results.")
    parser.add_argument(
        "--output_csv",
        default="sweep_metrics.csv",
        help="Output CSV filename (written under results_root).",
    )
    args = parser.parse_args()

    rows = []
    for run_dir, result_path in _find_results_files(args.results_root):
        try:
            result = torch.load(result_path, map_location="cpu")
        except Exception as exc:
            print(f"Skip {result_path}: {exc}")
            continue

        args_yaml = _load_args_yaml(os.path.join(run_dir, "args_inference.yaml"))
        run_tag = os.path.basename(os.path.dirname(run_dir))
        seed = os.path.basename(run_dir)

        metrics = getattr(result, "metrics", None)
        t_generator = _to_float(getattr(result, "t_generator", None))
        t_guide = _to_float(getattr(result, "t_guide", None))
        t_inference_total = _to_float(getattr(result, "t_inference_total", None))

        success = _to_float(getattr(getattr(metrics, "trajs_all", None), "success", None))
        success_no_joint_limits_vel_acc = _to_float(
            getattr(getattr(metrics, "trajs_all", None), "success_no_joint_limits_vel_acc", None)
        )
        fraction_valid = _to_float(getattr(getattr(metrics, "trajs_all", None), "fraction_valid", None))
        fraction_valid_no_joint_limits_vel_acc = _to_float(
            getattr(getattr(metrics, "trajs_all", None), "fraction_valid_no_joint_limits_vel_acc", None)
        )
        collision_intensity = _to_float(getattr(getattr(metrics, "trajs_all", None), "collision_intensity", None))

        ee_pose_goal_error_position_norm_best = _to_float(
            getattr(getattr(metrics, "trajs_best", None), "ee_pose_goal_error_position_norm", None)
        )
        ee_pose_goal_error_orientation_norm_best = _to_float(
            getattr(getattr(metrics, "trajs_best", None), "ee_pose_goal_error_orientation_norm", None)
        )

        ee_pose_goal_error_position_norm_mean_all = _to_float(
            getattr(getattr(metrics, "trajs_all", None), "ee_pose_goal_error_position_norm_mean", None)
        )
        ee_pose_goal_error_position_norm_std_all = _to_float(
            getattr(getattr(metrics, "trajs_all", None), "ee_pose_goal_error_position_norm_std", None)
        )
        ee_pose_goal_error_orientation_norm_mean_all = _to_float(
            getattr(getattr(metrics, "trajs_all", None), "ee_pose_goal_error_orientation_norm_mean", None)
        )
        ee_pose_goal_error_orientation_norm_std_all = _to_float(
            getattr(getattr(metrics, "trajs_all", None), "ee_pose_goal_error_orientation_norm_std", None)
        )

        ee_pose_goal_error_position_norm_mean_valid = _to_float(
            getattr(getattr(metrics, "trajs_valid", None), "ee_pose_goal_error_position_norm_mean", None)
        )
        ee_pose_goal_error_position_norm_std_valid = _to_float(
            getattr(getattr(metrics, "trajs_valid", None), "ee_pose_goal_error_position_norm_std", None)
        )
        ee_pose_goal_error_orientation_norm_mean_valid = _to_float(
            getattr(getattr(metrics, "trajs_valid", None), "ee_pose_goal_error_orientation_norm_mean", None)
        )
        ee_pose_goal_error_orientation_norm_std_valid = _to_float(
            getattr(getattr(metrics, "trajs_valid", None), "ee_pose_goal_error_orientation_norm_std", None)
        )

        path_best = _to_float(getattr(getattr(metrics, "trajs_best", None), "path_length", None))
        path_valid_mean = _to_float(getattr(getattr(metrics, "trajs_valid", None), "path_length_mean", None))
        smoothness_best = _to_float(getattr(getattr(metrics, "trajs_best", None), "smoothness", None))

        row = {
            "run_tag": run_tag,
            "seed": seed,
            "result_file": os.path.basename(result_path),
            "success": success,
            "success_no_joint_limits_vel_acc": success_no_joint_limits_vel_acc,
            "fraction_valid": fraction_valid,
            "fraction_valid_no_joint_limits_vel_acc": fraction_valid_no_joint_limits_vel_acc,
            "collision_intensity": collision_intensity,
            "ee_pose_goal_error_position_norm_best": ee_pose_goal_error_position_norm_best,
            "ee_pose_goal_error_orientation_norm_best": ee_pose_goal_error_orientation_norm_best,
            "ee_pose_goal_error_position_norm_mean_all": ee_pose_goal_error_position_norm_mean_all,
            "ee_pose_goal_error_position_norm_std_all": ee_pose_goal_error_position_norm_std_all,
            "ee_pose_goal_error_orientation_norm_mean_all": ee_pose_goal_error_orientation_norm_mean_all,
            "ee_pose_goal_error_orientation_norm_std_all": ee_pose_goal_error_orientation_norm_std_all,
            "ee_pose_goal_error_position_norm_mean_valid": ee_pose_goal_error_position_norm_mean_valid,
            "ee_pose_goal_error_position_norm_std_valid": ee_pose_goal_error_position_norm_std_valid,
            "ee_pose_goal_error_orientation_norm_mean_valid": ee_pose_goal_error_orientation_norm_mean_valid,
            "ee_pose_goal_error_orientation_norm_std_valid": ee_pose_goal_error_orientation_norm_std_valid,
            "path_length_best": path_best,
            "path_length_valid_mean": path_valid_mean,
            "smoothness_best": smoothness_best,
            "t_inference_total": t_inference_total,
            "t_generator": t_generator,
            "t_guide": t_guide,
            "diffusion_sampling_method": _safe_get(args_yaml, "diffusion_sampling_method", ""),
            "planner_alg": _safe_get(args_yaml, "planner_alg", ""),
        }
        rows.append(row)

    if not rows:
        print("No results found.")
        return

    output_csv = os.path.join(args.results_root, args.output_csv)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")

    # Rankings for success (desc), fraction_valid (desc), path length (asc), speed (asc)
    rank_success = sorted(
        rows,
        key=lambda r: (
            -(-1.0 if _is_nan(r["success"]) else r["success"]),
            float("inf") if _is_nan(r["t_inference_total"]) else r["t_inference_total"],
        ),
    )
    rank_fraction_valid = sorted(
        rows,
        key=lambda r: (
            -(-1.0 if _is_nan(r["fraction_valid"]) else r["fraction_valid"]),
            float("inf") if _is_nan(r["t_inference_total"]) else r["t_inference_total"],
        ),
    )
    rank_path = sorted(
        rows,
        key=lambda r: (
            r["path_length_best"] if r["path_length_best"] == r["path_length_best"] else float("inf")
        ),
    )
    rank_speed = sorted(
        rows,
        key=lambda r: (
            float("inf") if _is_nan(r["t_inference_total"]) else r["t_inference_total"],
        ),
    )

    def _write_rank(name: str, data: List[Dict[str, Any]]) -> None:
        path = os.path.join(args.results_root, name)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        print(f"Wrote {path}")

    _write_rank("sweep_rank_success.csv", rank_success)
    _write_rank("sweep_rank_fraction_valid.csv", rank_fraction_valid)
    _write_rank("sweep_rank_path_length.csv", rank_path)
    _write_rank("sweep_rank_speed.csv", rank_speed)


if __name__ == "__main__":
    main()
