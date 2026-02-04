from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from dmeval.adapters.base import TrialArtifact
from dmeval.utils import as_plain_mapping, flatten_mapping, to_float


class MpdResultsAdapter:
    name = "mpd"

    def discover_trials(self, results_root: Path) -> Iterable[TrialArtifact]:
        for root, _, files in os.walk(results_root):
            for fname in files:
                if not (fname.startswith("results_single_plan-") and fname.endswith(".pt")):
                    continue
                result_path = Path(root) / fname
                run_dir = result_path.parent
                rel = run_dir.relative_to(results_root)
                parts = rel.parts

                # Expected layouts:
                # - <results_root>/<run_tag>/<seed>/results_single_plan-*.pt
                # - <results_root>/<scenario>/<run_tag>/<seed>/results_single_plan-*.pt
                scenario = ""
                if len(parts) >= 3:
                    scenario = str(parts[-3])
                elif len(parts) >= 2:
                    scenario = str(results_root.name)

                run_tag = parts[-2] if len(parts) >= 2 else parts[0] if parts else ""
                seed = parts[-1] if len(parts) >= 2 else ""

                yield TrialArtifact(
                    results_root=results_root,
                    scenario=scenario,
                    run_dir=run_dir,
                    result_path=result_path,
                    run_tag=str(run_tag),
                    seed=str(seed),
                    result_file=fname,
                )

    def load_trial_object(self, artifact: TrialArtifact) -> Any:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Loading MPD .pt results requires torch. Install dmeval with the 'mpd' extra: dmeval[mpd]."
            ) from exc

        return torch.load(str(artifact.result_path), map_location="cpu")

    def _load_args_inference(self, run_dir: Path) -> Dict[str, Any]:
        path = run_dir / "args_inference.yaml"
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text()) or {}

    def _infer_scenario_label(self, *, results_root: Path, args_inference: Dict[str, Any]) -> str:
        env_id_replace = args_inference.get("env_id_replace", None)
        if env_id_replace not in (None, False):
            token = str(env_id_replace).strip().lower()
            if token and token not in ("none", "null", "false"):
                return "scene02"
        return "scene02" if "replace" in str(results_root).lower() else "scene01"

    def _safe_get(self, d: Dict[str, Any], key: str, default: Any = "") -> Any:
        return d[key] if key in d else default

    def _extract_known_metrics(self, metrics: Any) -> Dict[str, Any]:
        trajs_all = getattr(metrics, "trajs_all", None)
        trajs_valid = getattr(metrics, "trajs_valid", None)
        trajs_best = getattr(metrics, "trajs_best", None)

        return {
            "success": to_float(getattr(trajs_all, "success", None)),
            "success_no_joint_limits_vel_acc": to_float(
                getattr(trajs_all, "success_no_joint_limits_vel_acc", None)
            ),
            "fraction_valid": to_float(getattr(trajs_all, "fraction_valid", None)),
            "fraction_valid_no_joint_limits_vel_acc": to_float(
                getattr(trajs_all, "fraction_valid_no_joint_limits_vel_acc", None)
            ),
            "collision_intensity": to_float(getattr(trajs_all, "collision_intensity", None)),
            "ee_pose_goal_error_position_norm_best": to_float(
                getattr(trajs_best, "ee_pose_goal_error_position_norm", None)
            ),
            "ee_pose_goal_error_orientation_norm_best": to_float(
                getattr(trajs_best, "ee_pose_goal_error_orientation_norm", None)
            ),
            "ee_pose_goal_error_position_norm_mean_all": to_float(
                getattr(trajs_all, "ee_pose_goal_error_position_norm_mean", None)
            ),
            "ee_pose_goal_error_position_norm_std_all": to_float(
                getattr(trajs_all, "ee_pose_goal_error_position_norm_std", None)
            ),
            "ee_pose_goal_error_orientation_norm_mean_all": to_float(
                getattr(trajs_all, "ee_pose_goal_error_orientation_norm_mean", None)
            ),
            "ee_pose_goal_error_orientation_norm_std_all": to_float(
                getattr(trajs_all, "ee_pose_goal_error_orientation_norm_std", None)
            ),
            "ee_pose_goal_error_position_norm_mean_valid": to_float(
                getattr(trajs_valid, "ee_pose_goal_error_position_norm_mean", None)
            ),
            "ee_pose_goal_error_position_norm_std_valid": to_float(
                getattr(trajs_valid, "ee_pose_goal_error_position_norm_std", None)
            ),
            "ee_pose_goal_error_orientation_norm_mean_valid": to_float(
                getattr(trajs_valid, "ee_pose_goal_error_orientation_norm_mean", None)
            ),
            "ee_pose_goal_error_orientation_norm_std_valid": to_float(
                getattr(trajs_valid, "ee_pose_goal_error_orientation_norm_std", None)
            ),
            "path_length_best": to_float(getattr(trajs_best, "path_length", None)),
            "path_length_valid_mean": to_float(getattr(trajs_valid, "path_length_mean", None)),
            "smoothness_best": to_float(getattr(trajs_best, "smoothness", None)),
            "diversity_valid": to_float(getattr(trajs_valid, "diversity", None)),
        }

    def extract_row(
        self,
        artifact: TrialArtifact,
        trial_object: Any,
        *,
        include_all_config: bool,
    ) -> Dict[str, Any]:
        args_inference = self._load_args_inference(artifact.run_dir)
        scenario = artifact.scenario or self._infer_scenario_label(
            results_root=artifact.results_root, args_inference=args_inference
        )

        metrics = getattr(trial_object, "metrics", None)

        row: Dict[str, Any] = {
            "scenario": scenario,
            "env_id_replace": self._safe_get(args_inference, "env_id_replace", ""),
            "run_tag": artifact.run_tag,
            "seed": artifact.seed,
            "result_file": artifact.result_file,
            "t_inference_total": to_float(getattr(trial_object, "t_inference_total", None)),
            "t_generator": to_float(getattr(trial_object, "t_generator", None)),
            "t_guide": to_float(getattr(trial_object, "t_guide", None)),
            "diffusion_sampling_method": self._safe_get(args_inference, "diffusion_sampling_method", ""),
            "planner_alg": self._safe_get(args_inference, "planner_alg", ""),
        }

        if metrics is not None:
            row.update(self._extract_known_metrics(metrics))

        if include_all_config:
            flat_cfg = flatten_mapping(args_inference, prefix="cfg")
            # Avoid overwriting the core fields if names collide.
            for k, v in flat_cfg.items():
                if k in row:
                    continue
                row[k] = v

        # Optional: also include every numeric metric field we can find (future-proofing).
        metrics_map = as_plain_mapping(metrics)
        if metrics_map:
            flat_metrics = flatten_mapping(metrics_map, prefix="metrics")
            for k, v in flat_metrics.items():
                if k in row:
                    continue
                fv = to_float(v)
                if fv == fv:  # not NaN
                    row[k] = fv

        return row
