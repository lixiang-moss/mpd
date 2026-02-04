"""
Stage I：tune（调参阶段）。

职责（对应 `工具描述文档.md`）：
- 对每个 sampler 生成候选 candidates（grid/random/optuna）
- 串行运行被测系统：candidate × seed（每次 run 内含 N trials）
- Collect：抽取 trial 行 → 聚合为 run/candidate 指标
- Objective：阈值过滤 + 排序/评分，选择 best/top-k
- 输出 best_configs/<sampler>/{best_patch.yaml,best.yaml,topk.yaml} + tune_manifest.yaml

重要约束：
- 只串行（不引入并行调度）
- 不重算指标，只抽取与聚合
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import MPDAdapter
from .collect import aggregate_mean_std, to_dataframe, write_stage_metrics
from .objective import select_topk
from .runner import run_planner_subprocess
from .search import generate_candidates
from .util import cfg_to_yaml_str, ensure_dir, ensure_empty_dir, merge_patches, read_yaml, write_text, write_yaml


def run_tune(cfg: Any) -> None:
    """执行 Stage I（tune）。"""
    if not bool(getattr(cfg.tune, "enabled", True)):
        return

    # pipeline.root 是所有输出的根目录；默认拒绝覆盖，避免实验结果被意外抹掉。
    pipeline_root = Path(str(cfg.pipeline.root))
    allow_overwrite = bool(getattr(cfg.pipeline, "allow_overwrite", False))
    ensure_empty_dir(pipeline_root, allow_overwrite=allow_overwrite)

    stage_dir = pipeline_root / str(getattr(cfg.tune, "stage_dir", "tune"))
    ensure_dir(stage_dir)

    # Stage I 产出会写入 best_configs，供 Stage II 复用。
    best_root = pipeline_root / "best_configs"
    ensure_dir(best_root)

    adapter = _build_adapter(cfg)

    scenario_name = str(cfg.tune.scenario.name)
    base_cfg_path = Path(str(cfg.tune.scenario.base_cfg))
    seeds = [int(s) for s in list(cfg.tune.seeds)]
    top_k = int(getattr(cfg.tune, "top_k", 1))

    manifest: dict[str, Any] = {
        "tool": "dmeval",
        "stage": "tune",
        "pipeline_root": str(pipeline_root),
        "scenario": {"name": scenario_name, "base_cfg": str(base_cfg_path)},
        "seeds": seeds,
        "common_inference_args": {
            "selection_start_goal": str(cfg.common_inference_args.selection_start_goal),
            "n_start_goal_states": int(cfg.common_inference_args.n_start_goal_states),
            "save_results_single_plan_low_mem": bool(cfg.common_inference_args.save_results_single_plan_low_mem),
            "device": str(cfg.common_inference_args.device),
        },
        "objective": _to_plain(cfg.objective),
        "samplers": [],
        "resolved_config_yaml": cfg_to_yaml_str(cfg),
    }

    # “一键调优多个采样器”：按 tune.samplers 顺序串行跑完。
    for sampler_cfg in list(cfg.tune.samplers):
        sampler_name = str(sampler_cfg.name)
        sampler_dir = stage_dir / sampler_name
        ensure_dir(sampler_dir)
        candidates_dir = sampler_dir / "candidates"
        ensure_dir(candidates_dir)

        base_patch = _to_plain(getattr(sampler_cfg, "patch_base", {})) or {}
        search_cfg = getattr(sampler_cfg, "search", None)
        candidates = generate_candidates(base_patch=base_patch, search_cfg=search_cfg)

        # trial_rows 是“每个 trial 一行”的统一口径记录（来自 Adapter）。
        trial_rows: list[dict[str, Any]] = []

        for cand in candidates:
            cand_dir = candidates_dir / cand.candidate_id
            ensure_dir(cand_dir)

            # 1) 把 patch 落盘，便于复现与调试
            patch_path = cand_dir / "patch.yaml"
            write_yaml(patch_path, cand.patch)

            # 2) 生成本次 candidate 的推理配置：base_cfg + patch
            merged_cfg = read_yaml(base_cfg_path)
            merged_cfg = merge_patches(merged_cfg, cand.patch)
            cfg_path = cand_dir / "cfg_inference.yaml"
            write_yaml(cfg_path, merged_cfg)

            # 3) results_dir 由 DMEval 管理；planner 可能会在里面创建 seed 子目录。
            results_dir = cand_dir / "results"
            ensure_dir(results_dir)

            # 4) 串行运行：candidate × seed
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
                    log_path=cand_dir / f"seed{seed}.log",
                )

            # 5) 抽取产物：优先 jsonl，否则回退 pt（由 Adapter 决定）
            run_tag = f"{sampler_name}__{cand.candidate_id}"
            rows = adapter.collect_trial_rows(
                results_root=results_dir,
                scenario=scenario_name,
                run_tag=run_tag,
                sampler=sampler_name,
                candidate_id=cand.candidate_id,
            )
            trial_rows.extend(rows)

        # 聚合层次（与论文/规格一致）：
        # - trial_df：每个 trial 一行
        # - run_df：按 seed 聚合（candidate × seed）
        # - candidate_df：跨 seed 聚合（用于 objective 选 best）
        trial_df = to_dataframe(trial_rows)
        run_df = aggregate_mean_std(df=trial_df, group_cols=["scenario", "sampler", "candidate_id", "run_tag", "seed"])
        candidate_df = aggregate_mean_std(df=trial_df, group_cols=["scenario", "sampler", "candidate_id"])

        write_stage_metrics(
            out_dir=sampler_dir,
            trial_df=trial_df,
            run_df=run_df,
            extra={"candidate_metrics": candidate_df},
        )

        # objective 选出 top-k 与 best
        topk_df, best_row = select_topk(
            candidate_df=candidate_df,
            objective_cfg=cfg.objective,
            top_k=top_k,
            id_field="candidate_id",
        )
        topk_records = topk_df.sort_values("candidate_id").to_dict(orient="records") if not topk_df.empty else []

        sampler_best_dir = best_root / sampler_name
        ensure_dir(sampler_best_dir)
        write_yaml(sampler_best_dir / "topk.yaml", topk_records)

        best_patch: dict[str, Any] | None = None
        if best_row is not None:
            best_candidate_id = str(best_row.get("candidate_id", ""))
            best_candidate_dir = candidates_dir / best_candidate_id
            best_patch = read_yaml(best_candidate_dir / "patch.yaml")

            # best_patch.yaml 是 Stage II 的核心输入；best.yaml 只是为了方便检查（完整 cfg）。
            write_yaml(sampler_best_dir / "best_patch.yaml", best_patch)
            best_full = merge_patches(read_yaml(base_cfg_path), best_patch)
            write_yaml(sampler_best_dir / "best.yaml", best_full)

        manifest["samplers"].append(
            {
                "name": sampler_name,
                "base_patch": base_patch,
                "search": _to_plain(search_cfg),
                "n_candidates": len(candidates),
                "best": best_row,
                "best_configs_dir": str(sampler_best_dir),
            }
        )

    # manifest 记录了本次运行的“全量配置/搜索空间/阈值等”，用于论文复现与追溯。
    write_yaml(pipeline_root / "tune_manifest.yaml", manifest)
    write_text(pipeline_root / "DONE_TUNE.txt", "OK\n")


def _build_adapter(cfg: Any) -> MPDAdapter:
    """
    构造 Adapter。

    当前 L1 只实现了 MPDAdapter（首个适配器）；未来扩展其它 planner 时建议在此处扩展分发逻辑。
    """
    adapter_type = str(getattr(cfg.adapter, "type", "mpd")).lower()
    if adapter_type != "mpd":
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    return MPDAdapter.from_cfg(cfg.adapter)


def _to_plain(value: Any) -> Any:
    """把 OmegaConf 容器转成普通 Python dict/list（用于写 manifest）。"""
    try:
        # OmegaConf containers
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value
