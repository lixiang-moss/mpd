"""
MPDAdapter: parse MPD artifact directories into unified per-trial row dicts.

Why does a "general evaluation tool" contain an MPD-specific adapter?
- The evaluation workflow (tune/compare/aggregation/ranking) is general, but output formats differ per planner
- Therefore "results directory parsing" must live behind an Adapter interface
- MPDAdapter is the first concrete adapter and also serves as a reference implementation

Input (MPD artifacts, per spec):
- Each seed directory contains at least:
  - args_inference.yaml
  - results_single_plan-XXX.pt (N files)
- Recommended lightweight output (preferred when present):
  - trial_metrics.jsonl (one JSON object per trial)

Output (DMEval side, per-trial row dicts):
- metadata: scenario/run_tag/seed/trial_id/sampler/candidate_id, etc.
- metric fields: success/fraction_valid/collision_intensity/time/path_length/... (aligned with the spec where possible)
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


def _to_float(value: Any) -> float:
    """Convert various numeric-like values (including torch tensors) to float; return NaN on failure."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        import torch  # type: ignore

        if hasattr(torch, "is_tensor") and torch.is_tensor(value):
            return float(value.detach().cpu().item())
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML; return an empty dict if the file does not exist."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


_TRIAL_ID_RE = re.compile(r"results_single_plan-(\d+)\.pt$")


@dataclass(frozen=True)
class MPDAdapterConfig:
    """Config options for MPDAdapter (filenames, priority, etc.)."""
    prefer_jsonl: bool = True
    trial_metrics_jsonl: str = "trial_metrics.jsonl"
    run_summary_yaml: str = "run_summary.yaml"
    args_inference_yaml: str = "args_inference.yaml"
    results_pt_glob: str = "results_single_plan-*.pt"


class MPDAdapter:
    """Adapter for parsing MPD results."""
    def __init__(self, cfg: MPDAdapterConfig):
        self.cfg = cfg

    @classmethod
    def from_cfg(cls, adapter_cfg: Any) -> "MPDAdapter":
        """Construct `MPDAdapterConfig` from a Hydra config node."""
        cfg = MPDAdapterConfig(
            prefer_jsonl=bool(getattr(adapter_cfg, "prefer_jsonl", True)),
            trial_metrics_jsonl=str(getattr(adapter_cfg, "trial_metrics_jsonl", "trial_metrics.jsonl")),
            run_summary_yaml=str(getattr(adapter_cfg, "run_summary_yaml", "run_summary.yaml")),
            args_inference_yaml=str(getattr(adapter_cfg, "args_inference_yaml", "args_inference.yaml")),
            results_pt_glob=str(getattr(adapter_cfg, "results_pt_glob", "results_single_plan-*.pt")),
        )
        return cls(cfg)

    def collect_trial_rows(
        self,
        *,
        results_root: Path,
        scenario: str,
        run_tag: str,
        sampler: str | None = None,
        candidate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Collect per-trial rows from a run results root.

        Args:
          results_root: results directory for one run (passed by DMEval, e.g. `<candidate_dir>/results`)
          scenario/run_tag: metadata used in CSV outputs
          sampler/candidate_id: optional metadata (filled in Stage I)

        Behavior:
          - When `prefer_jsonl=true` and JSONL exists: parse JSONL first (lightweight, no torch required)
          - Otherwise: fall back to parsing `results_single_plan-*.pt` (requires torch)
        """
        results_root = results_root.resolve()
        rows: list[dict[str, Any]] = []

        if self.cfg.prefer_jsonl:
            jsonl_files = self._find_files(results_root, self.cfg.trial_metrics_jsonl)
            if jsonl_files:
                for jsonl_path in jsonl_files:
                    seed_dir = jsonl_path.parent
                    seed = seed_dir.name
                    args_inf = _load_yaml(seed_dir / self.cfg.args_inference_yaml)
                    # Each JSONL line is one trial-metrics dict; we only add metadata and merge fields.
                    rows.extend(
                        self._rows_from_jsonl(
                            jsonl_path=jsonl_path,
                            scenario=scenario,
                            run_tag=run_tag,
                            seed=seed,
                            args_inference=args_inf,
                            sampler=sampler,
                            candidate_id=candidate_id,
                        )
                    )
                return rows

        # Fallback: scan .pt results and parse via torch.load.
        pt_files = self._find_glob(results_root, self.cfg.results_pt_glob)
        for pt_path in pt_files:
            seed_dir = pt_path.parent
            seed = seed_dir.name
            args_inf = _load_yaml(seed_dir / self.cfg.args_inference_yaml)
            trial_id = self._infer_trial_id(pt_path)
            row = self._row_from_pt(
                pt_path=pt_path,
                scenario=scenario,
                run_tag=run_tag,
                seed=seed,
                trial_id=trial_id,
                args_inference=args_inf,
                sampler=sampler,
                candidate_id=candidate_id,
            )
            rows.append(row)

        return rows

    def _find_files(self, root: Path, filename: str) -> list[Path]:
        """Recursively find files by exact filename (used for JSONL/args_inference.yaml)."""
        hits: list[Path] = []
        for r, _, files in os.walk(root):
            if filename in files:
                hits.append(Path(r) / filename)
        return sorted(hits)

    def _find_glob(self, root: Path, pattern: str) -> list[Path]:
        """Recursively find files by glob-like pattern (used for results_single_plan-*.pt)."""
        hits: list[Path] = []
        for r, _, files in os.walk(root):
            for fname in files:
                if Path(fname).match(pattern):
                    hits.append(Path(r) / fname)
        return sorted(hits)

    def _infer_trial_id(self, path: Path) -> int:
        """Extract `trial_id` from `results_single_plan-XXX.pt` filename."""
        m = _TRIAL_ID_RE.search(path.name)
        return int(m.group(1)) if m else -1

    def _rows_from_jsonl(
        self,
        *,
        jsonl_path: Path,
        scenario: str,
        run_tag: str,
        seed: str,
        args_inference: dict[str, Any],
        sampler: str | None,
        candidate_id: str | None,
    ) -> Iterable[dict[str, Any]]:
        """Parse `trial_metrics.jsonl` and add metadata fields."""
        with jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                trial_id = int(obj.get("trial_id", i))
                # base_meta provides scenario/run_tag/seed/trial_id/planner_alg/diffusion_sampling_method, etc.
                row = self._base_meta(
                    scenario=scenario,
                    run_tag=run_tag,
                    seed=seed,
                    trial_id=trial_id,
                    args_inference=args_inference,
                    sampler=sampler,
                    candidate_id=candidate_id,
                )
                row.update(obj)
                yield row

    def _row_from_pt(
        self,
        *,
        pt_path: Path,
        scenario: str,
        run_tag: str,
        seed: str,
        trial_id: int,
        args_inference: dict[str, Any],
        sampler: str | None,
        candidate_id: str | None,
    ) -> dict[str, Any]:
        """
        Parse a single `.pt` trial file (MPD default artifact).

        Notes:
        - Strictly follows "extract only, do not recompute": reads fields from `result.metrics` only.
        - Torch is imported only in this branch so JSONL-only mode does not require torch.
        """
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "torch is required to parse MPD .pt results; install torch or enable trial_metrics.jsonl output."
            ) from exc

        result = torch.load(str(pt_path), map_location="cpu")
        metrics = getattr(result, "metrics", None)

        row = self._base_meta(
            scenario=scenario,
            run_tag=run_tag,
            seed=seed,
            trial_id=trial_id,
            args_inference=args_inference,
            sampler=sampler,
            candidate_id=candidate_id,
        )

        row.update(
            {
                # time
                "t_inference_total": _to_float(getattr(result, "t_inference_total", None)),
                "t_generator": _to_float(getattr(result, "t_generator", None)),
                "t_guide": _to_float(getattr(result, "t_guide", None)),
                # feasibility/safety (trajs_all)
                "success": _to_float(getattr(getattr(metrics, "trajs_all", None), "success", None)),
                "fraction_valid": _to_float(getattr(getattr(metrics, "trajs_all", None), "fraction_valid", None)),
                "collision_intensity": _to_float(
                    getattr(getattr(metrics, "trajs_all", None), "collision_intensity", None)
                ),
                # accuracy
                "ee_pose_goal_error_position_norm_best": _to_float(
                    getattr(getattr(metrics, "trajs_best", None), "ee_pose_goal_error_position_norm", None)
                ),
                "ee_pose_goal_error_orientation_norm_best": _to_float(
                    getattr(getattr(metrics, "trajs_best", None), "ee_pose_goal_error_orientation_norm", None)
                ),
                "ee_pose_goal_error_position_norm_mean_all": _to_float(
                    getattr(getattr(metrics, "trajs_all", None), "ee_pose_goal_error_position_norm_mean", None)
                ),
                "ee_pose_goal_error_position_norm_std_all": _to_float(
                    getattr(getattr(metrics, "trajs_all", None), "ee_pose_goal_error_position_norm_std", None)
                ),
                "ee_pose_goal_error_orientation_norm_mean_all": _to_float(
                    getattr(getattr(metrics, "trajs_all", None), "ee_pose_goal_error_orientation_norm_mean", None)
                ),
                "ee_pose_goal_error_orientation_norm_std_all": _to_float(
                    getattr(getattr(metrics, "trajs_all", None), "ee_pose_goal_error_orientation_norm_std", None)
                ),
                "ee_pose_goal_error_position_norm_mean_valid": _to_float(
                    getattr(getattr(metrics, "trajs_valid", None), "ee_pose_goal_error_position_norm_mean", None)
                ),
                "ee_pose_goal_error_position_norm_std_valid": _to_float(
                    getattr(getattr(metrics, "trajs_valid", None), "ee_pose_goal_error_position_norm_std", None)
                ),
                "ee_pose_goal_error_orientation_norm_mean_valid": _to_float(
                    getattr(getattr(metrics, "trajs_valid", None), "ee_pose_goal_error_orientation_norm_mean", None)
                ),
                "ee_pose_goal_error_orientation_norm_std_valid": _to_float(
                    getattr(getattr(metrics, "trajs_valid", None), "ee_pose_goal_error_orientation_norm_std", None)
                ),
                # trajectory quality
                "path_length_best": _to_float(getattr(getattr(metrics, "trajs_best", None), "path_length", None)),
                "path_length_valid_mean": _to_float(
                    getattr(getattr(metrics, "trajs_valid", None), "path_length_mean", None)
                ),
                "path_length_valid_std": _to_float(
                    getattr(getattr(metrics, "trajs_valid", None), "path_length_std", None)
                ),
                "smoothness_best": _to_float(getattr(getattr(metrics, "trajs_best", None), "smoothness", None)),
                "smoothness_valid_mean": _to_float(
                    getattr(getattr(metrics, "trajs_valid", None), "smoothness_mean", None)
                ),
                "smoothness_valid_std": _to_float(
                    getattr(getattr(metrics, "trajs_valid", None), "smoothness_std", None)
                ),
                # diversity (may be missing / NaN)
                "diversity": _to_float(
                    getattr(getattr(getattr(metrics, "trajs_valid", None), "diversity", None), "value", None)
                ),
            }
        )

        # Compatibility: some versions may store `diversity` as a number/tensor rather than an object with `.value`.
        if math.isnan(row["diversity"]):
            row["diversity"] = _to_float(getattr(getattr(metrics, "trajs_valid", None), "diversity", None))

        return row

    def _base_meta(
        self,
        *,
        scenario: str,
        run_tag: str,
        seed: str,
        trial_id: int,
        args_inference: dict[str, Any],
        sampler: str | None,
        candidate_id: str | None,
    ) -> dict[str, Any]:
        """
        Construct metadata fields required for every row.

        These fields are not part of MPD's metric computation, but are crucial for DMEval aggregation/ranking/traceability:
        - scenario / run_tag / seed / trial_id
        - sampler / candidate_id (used to organize Stage I/II comparisons)
        """
        row: dict[str, Any] = {
            "scenario": scenario,
            "run_tag": run_tag,
            "seed": seed,
            "trial_id": trial_id,
            "planner_alg": args_inference.get("planner_alg", ""),
            "diffusion_sampling_method": args_inference.get("diffusion_sampling_method", ""),
        }
        if sampler is not None:
            row["sampler"] = sampler
        if candidate_id is not None:
            row["candidate_id"] = candidate_id
        # Optional meta passthrough (helpful for disambiguating scenarios)
        if "env_id_replace" in args_inference:
            row["env_id_replace"] = args_inference.get("env_id_replace")
        return row
