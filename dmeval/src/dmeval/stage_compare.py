"""
Stage II：compare（公平对比阶段）。

职责（对应 `工具描述文档.md`）：
- 读取 Stage I 产出的 best_configs/<sampler>/best_patch.yaml
- 对每个 scenario base_cfg 打 patch，生成最终 cfg_inference.yaml
- 串行运行被测系统：scenario × sampler × seed
- Collect：抽取 trial 行 → 聚合为 run/agg 指标
- Rank + Plot：输出排名 CSV 与基础图表

重要约束：
- 只串行（不引入并行调度）
- 不重算指标，只抽取与聚合
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import shutil

from .adapters import MPDAdapter
from .collect import aggregate_mean_std, to_dataframe, write_stage_metrics, write_csv
from .plotting import plot_stage_compare
from .runner import run_planner_subprocess
from .util import cfg_to_yaml_str, ensure_dir, merge_patches, read_yaml, write_text, write_yaml


def run_compare(cfg: Any) -> None:
    """执行 Stage II（compare）。"""
    if not bool(getattr(cfg.compare, "enabled", True)):
        return

    pipeline_root = Path(str(cfg.pipeline.root))
    if not pipeline_root.exists():
        raise FileNotFoundError(
            f"pipeline.root does not exist: {pipeline_root}. Run `dmeval tune` first or set the correct pipeline.root."
        )

    stage_dir = pipeline_root / str(getattr(cfg.compare, "stage_dir", "compare"))
    allow_overwrite = bool(getattr(cfg.pipeline, "allow_overwrite", False))
    if stage_dir.exists():
        if not allow_overwrite:
            raise FileExistsError(f"Refuse to overwrite existing compare directory: {stage_dir}")
        # compare 目录属于“派生产物”，允许在 allow_overwrite=true 时重建。
        shutil.rmtree(stage_dir)
    ensure_dir(stage_dir)

    adapter = _build_adapter(cfg)

    best_root = Path(str(cfg.compare.best_configs_root))
    if not best_root.exists():
        raise FileNotFoundError(f"best_configs_root not found: {best_root}")

    seeds = [int(s) for s in list(cfg.compare.seeds)]
    scenarios = list(cfg.compare.scenarios)
    sampler_names = _discover_samplers(best_root)
    if not sampler_names:
        raise RuntimeError(f"No samplers found under best_configs_root: {best_root}")

    # trial_rows 仍然是“每个 trial 一行”的统一口径记录（来自 Adapter）。
    trial_rows: list[dict[str, Any]] = []

    for scenario_cfg in scenarios:
        scenario_name = str(scenario_cfg.name)
        base_cfg_path = Path(str(scenario_cfg.base_cfg))

        for sampler_name in sampler_names:
            # Stage II 的核心输入：Stage I 固化的 best_patch.yaml
            patch_path = best_root / sampler_name / "best_patch.yaml"
            if not patch_path.exists():
                continue
            patch = read_yaml(patch_path)
            # 生成最终推理配置：scenario 的 base_cfg + sampler 的 best_patch
            merged_cfg = merge_patches(read_yaml(base_cfg_path), patch)

            run_dir = stage_dir / scenario_name / sampler_name
            ensure_dir(run_dir)
            cfg_path = run_dir / "cfg_inference.yaml"
            write_yaml(cfg_path, merged_cfg)

            results_dir = run_dir / "results"
            ensure_dir(results_dir)

            # 串行运行：scenario × sampler × seed
            for seed in seeds:
                run_planner_subprocess(
                    python=str(cfg.planner.python),
                    entrypoint=str(cfg.planner.entrypoint),
                    workdir=Path(str(cfg.planner.workdir)),
                    cfg_inference_path=cfg_path,
                    results_dir=results_dir,
                    seed=seed,
                    selection_start_goal=str(cfg.common_inference_args.selection_start_goal),
                    n_start_goal_states=int(cfg.common_inference_args.n_start_goal_states),
                    save_results_single_plan_low_mem=bool(cfg.common_inference_args.save_results_single_plan_low_mem),
                    device=str(cfg.common_inference_args.device),
                    extra_args=list(getattr(cfg.planner, "extra_args", []) or []),
                    log_path=run_dir / f"seed{seed}.log",
                )

            # 抽取产物：由 Adapter 决定读 jsonl 还是 pt
            run_tag = sampler_name
            rows = adapter.collect_trial_rows(
                results_root=results_dir,
                scenario=scenario_name,
                run_tag=run_tag,
                sampler=sampler_name,
                candidate_id=None,
            )
            trial_rows.extend(rows)

    # 聚合层次（与论文/规格一致）：
    # - run_df：按 seed 聚合（scenario × sampler × seed）
    # - run_agg_df：跨 seed 聚合（scenario × sampler），用于公平比较与绘图/排名
    trial_df = to_dataframe(trial_rows)
    run_df = aggregate_mean_std(df=trial_df, group_cols=["scenario", "sampler", "run_tag", "seed"])
    run_agg_df = aggregate_mean_std(df=trial_df, group_cols=["scenario", "sampler", "run_tag"])

    write_stage_metrics(out_dir=stage_dir, trial_df=trial_df, run_df=run_df, extra={"run_metrics_agg": run_agg_df})

    _write_rankings(run_agg_df, out_dir=stage_dir)
    plot_stage_compare(run_metrics_agg=run_agg_df, out_dir=stage_dir / "plots")

    manifest: dict[str, Any] = {
        "tool": "dmeval",
        "stage": "compare",
        "pipeline_root": str(pipeline_root),
        "best_configs_root": str(best_root),
        "seeds": seeds,
        "scenarios": [{"name": str(s.name), "base_cfg": str(s.base_cfg)} for s in scenarios],
        "samplers": sampler_names,
        "common_inference_args": {
            "selection_start_goal": str(cfg.common_inference_args.selection_start_goal),
            "n_start_goal_states": int(cfg.common_inference_args.n_start_goal_states),
            "save_results_single_plan_low_mem": bool(cfg.common_inference_args.save_results_single_plan_low_mem),
            "device": str(cfg.common_inference_args.device),
        },
        "resolved_config_yaml": cfg_to_yaml_str(cfg),
    }
    write_yaml(stage_dir / "compare_manifest.yaml", manifest)
    write_text(stage_dir / "DONE_COMPARE.txt", "OK\n")


def _build_adapter(cfg: Any) -> MPDAdapter:
    """构造 Adapter（当前 L1 只实现 MPDAdapter）。"""
    adapter_type = str(getattr(cfg.adapter, "type", "mpd")).lower()
    if adapter_type != "mpd":
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    return MPDAdapter.from_cfg(cfg.adapter)


def _discover_samplers(best_root: Path) -> list[str]:
    """从 best_configs_root 下发现有哪些 sampler（以存在 best_patch.yaml 为准）。"""
    out: list[str] = []
    if not best_root.exists():
        return out
    for p in best_root.iterdir():
        if p.is_dir() and (p / "best_patch.yaml").exists():
            out.append(p.name)
    return sorted(out)


def _write_rankings(run_agg_df: pd.DataFrame, *, out_dir: Path) -> None:
    """
    输出基础排名 CSV（满足规格里的最低要求）。

    排名口径（越靠前越好）：
    - success：success_mean 降序，t_inference_total_mean 升序
    - fraction_valid：fraction_valid_mean 降序，t_inference_total_mean 升序
    - path_length：path_length_best_mean 升序
    - speed：t_inference_total_mean 升序
    """
    if run_agg_df.empty:
        return

    def _safe(col: str, default: float) -> pd.Series:
        # 容错：列缺失/不可转数值时使用 default，保证排序逻辑不会崩。
        if col not in run_agg_df.columns:
            return pd.Series([default] * len(run_agg_df))
        s = pd.to_numeric(run_agg_df[col], errors="coerce").fillna(default)
        return s

    # 下面通过构造 _k1/_k2 做排序 key；越小越好。
    rank_success = run_agg_df.assign(
        _k1=-_safe("success_mean", -1.0),
        _k2=_safe("t_inference_total_mean", float("inf")),
    ).sort_values(["_k1", "_k2"])
    write_csv(rank_success.drop(columns=["_k1", "_k2"]), out_dir / "rank_success.csv")

    rank_fraction_valid = run_agg_df.assign(
        _k1=-_safe("fraction_valid_mean", -1.0),
        _k2=_safe("t_inference_total_mean", float("inf")),
    ).sort_values(["_k1", "_k2"])
    write_csv(rank_fraction_valid.drop(columns=["_k1", "_k2"]), out_dir / "rank_fraction_valid.csv")

    rank_path = run_agg_df.assign(
        _k1=_safe("path_length_best_mean", float("inf")),
    ).sort_values(["_k1"])
    write_csv(rank_path.drop(columns=["_k1"]), out_dir / "rank_path_length.csv")

    rank_speed = run_agg_df.assign(
        _k1=_safe("t_inference_total_mean", float("inf")),
    ).sort_values(["_k1"])
    write_csv(rank_speed.drop(columns=["_k1"]), out_dir / "rank_speed.csv")
