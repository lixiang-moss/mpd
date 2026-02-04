"""
MPDAdapter：把 MPD 的产物目录解析成“统一口径”的 trial 行字典。

为什么在“通用工具”里会有 MPD 专用 adapter？
- 通用的是评测流程（tune/compare/聚合/排名），不是所有 planner 的输出格式
- 因此必须把“结果目录解析”封装在 Adapter 层
- MPDAdapter 是第一个落地实现：既是适配器，也是参考实现

输入（MPD 侧产物，按 `工具描述文档.md`）：
- 每个 seed 目录至少包含：
  - args_inference.yaml
  - results_single_plan-XXX.pt（N 个）
- 推荐轻量输出（如果有则优先用）：
  - trial_metrics.jsonl（每个 trial 一行 JSON）

输出（DMEval 侧，trial 行 dict）：
- 元信息：scenario/run_tag/seed/trial_id/sampler/candidate_id 等
- 指标字段：success/fraction_valid/collision_intensity/time/path_length/...（字段口径尽量贴近规格）
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
    """把各种可能的数值类型（含 torch tensor）转换为 float；失败则返回 NaN。"""
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
    """读取 YAML；文件不存在则返回空 dict。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


_TRIAL_ID_RE = re.compile(r"results_single_plan-(\d+)\.pt$")


@dataclass(frozen=True)
class MPDAdapterConfig:
    """MPDAdapter 的可配置项（文件名/优先级等）。"""
    prefer_jsonl: bool = True
    trial_metrics_jsonl: str = "trial_metrics.jsonl"
    run_summary_yaml: str = "run_summary.yaml"
    args_inference_yaml: str = "args_inference.yaml"
    results_pt_glob: str = "results_single_plan-*.pt"


class MPDAdapter:
    """MPD 结果解析适配器。"""
    def __init__(self, cfg: MPDAdapterConfig):
        self.cfg = cfg

    @classmethod
    def from_cfg(cls, adapter_cfg: Any) -> "MPDAdapter":
        """从 Hydra 配置节点构造 MPDAdapterConfig。"""
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
        从一个结果根目录收集 trial 行。

        参数:
          results_root: 一个 run 的结果目录（DMEval 传入，例如 `<candidate_dir>/results`）
          scenario/run_tag: 用于写入 CSV 的元信息
          sampler/candidate_id: 可选元信息（Stage I 会填入）

        行为:
          - prefer_jsonl=true 且存在 jsonl 时：优先解析 jsonl（轻量、无需 torch）
          - 否则回退解析 results_single_plan-*.pt（需要 torch）
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
                    # jsonl 每一行就是一个 trial 的指标 dict，这里只补齐元信息并合并字段。
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

        # 回退分支：扫描 .pt 结果并用 torch.load 解析。
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
        """递归查找指定文件名（用于 jsonl/args_inference.yaml）。"""
        hits: list[Path] = []
        for r, _, files in os.walk(root):
            if filename in files:
                hits.append(Path(r) / filename)
        return sorted(hits)

    def _find_glob(self, root: Path, pattern: str) -> list[Path]:
        """递归按 glob pattern 查找（用于 results_single_plan-*.pt）。"""
        hits: list[Path] = []
        for r, _, files in os.walk(root):
            for fname in files:
                if Path(fname).match(pattern):
                    hits.append(Path(r) / fname)
        return sorted(hits)

    def _infer_trial_id(self, path: Path) -> int:
        """从 `results_single_plan-XXX.pt` 文件名提取 trial_id。"""
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
        """解析 trial_metrics.jsonl 并补齐元信息。"""
        with jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                trial_id = int(obj.get("trial_id", i))
                # base_meta 提供：scenario/run_tag/seed/trial_id/planner_alg/diffusion_sampling_method 等。
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
        解析单个 `.pt` trial 文件（MPD 默认产物）。

        说明：
        - 这里严格遵循“只抽取、不重算”原则：只读取 result.metrics 里的字段。
        - torch 依赖只在这个分支里导入，避免 DMEval 在 jsonl-only 模式下强依赖 torch。
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

        # 兼容性处理：有些版本可能把 diversity 直接存成数字/tensor，而不是对象.value。
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
        构造每行都需要的元信息字段。

        这些字段不参与 MPD 指标计算，但对 DMEval 的聚合/排名/可追溯性非常关键：
        - scenario / run_tag / seed / trial_id
        - sampler / candidate_id（Stage I/II 组织对比用）
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
