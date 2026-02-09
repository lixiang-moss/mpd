"""
Stage I: tune (hyperparameter tuning stage).

Responsibilities (per spec):
- Generate candidate patches for each sampler (grid/random/optuna reserved)
- Run the system-under-test serially: candidate × seed (each run contains N trials)
- Collect: extract trial rows → aggregate to run/candidate metrics
- Objective: constraints filtering + ranking/scoring to select best/top-k
- Output `best_configs/<sampler>/{best_patch.yaml,best.yaml,topk.yaml}` plus `tune_manifest.yaml`

Key constraints:
- Serial only (no parallel scheduling)
- No metric recomputation; extraction and aggregation only
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .adapters import MPDAdapter
from .collect import aggregate_mean_std, to_dataframe, write_stage_metrics
from .objective import select_topk
from .runner import run_planner_subprocess
from .search import generate_candidates
from .util import cfg_to_yaml_str, ensure_dir, ensure_empty_dir, merge_patches, read_yaml, write_text, write_yaml

_LINE = "-" * 72


def _log(msg: str) -> None:
    """Stage-I progress logging."""
    print(f"[dmeval:tune] {msg}", flush=True)


def _fmt_num(value: Any) -> str:
    """Format a number as a compact string (non-numerics are returned as-is)."""
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def _section(title: str) -> None:
    """Print a section divider for readability."""
    print(_LINE, flush=True)
    _log(title)
    print(_LINE, flush=True)


def run_tune(cfg: Any) -> None:
    """Run Stage I (tune)."""
    if not bool(getattr(cfg.tune, "enabled", True)):
        _log("skip (tune.enabled=false)")
        return

    started_all = time.monotonic()
    # `pipeline.root` is the root for all outputs; default is to refuse overwrites to protect results.
    pipeline_root = Path(str(cfg.pipeline.root))
    allow_overwrite = bool(getattr(cfg.pipeline, "allow_overwrite", False))
    ensure_empty_dir(pipeline_root, allow_overwrite=allow_overwrite)

    stage_dir = pipeline_root / str(getattr(cfg.tune, "stage_dir", "tune"))
    ensure_dir(stage_dir)

    # Stage-I outputs write to `best_configs/` for Stage II reuse.
    best_root = pipeline_root / "best_configs"
    ensure_dir(best_root)

    adapter = _build_adapter(cfg)

    scenario_name = str(cfg.tune.scenario.name)
    base_cfg_path = Path(str(cfg.tune.scenario.base_cfg))
    seeds = [int(s) for s in list(cfg.tune.seeds)]
    top_k = int(getattr(cfg.tune, "top_k", 1))
    samplers = list(cfg.tune.samplers)
    _section("STAGE I START")
    _log(
        f"start pipeline_root={pipeline_root} scenario={scenario_name} "
        f"samplers={len(samplers)} seeds={seeds} top_k={top_k}"
    )

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

    # "Tune multiple samplers in one go": run serially in the order of `tune.samplers`.
    for sampler_idx, sampler_cfg in enumerate(samplers, start=1):
        sampler_name = str(sampler_cfg.name)
        started_sampler = time.monotonic()
        sampler_dir = stage_dir / sampler_name
        ensure_dir(sampler_dir)
        candidates_dir = sampler_dir / "candidates"
        ensure_dir(candidates_dir)

        base_patch = _to_plain(getattr(sampler_cfg, "patch_base", {})) or {}
        search_cfg = getattr(sampler_cfg, "search", None)
        candidates = generate_candidates(base_patch=base_patch, search_cfg=search_cfg)
        _section(f"SAMPLER {sampler_idx}/{len(samplers)} | {sampler_name} | candidates={len(candidates)}")

        # `trial_rows` is the unified "one row per trial" record format (from the adapter).
        trial_rows: list[dict[str, Any]] = []

        for cand_idx, cand in enumerate(candidates, start=1):
            started_cand = time.monotonic()
            cand_dir = candidates_dir / cand.candidate_id
            ensure_dir(cand_dir)

            # 1) Persist patch for reproducibility/debugging
            patch_path = cand_dir / "patch.yaml"
            write_yaml(patch_path, cand.patch)

            # 2) Build this candidate's inference config: base_cfg + patch
            merged_cfg = read_yaml(base_cfg_path)
            merged_cfg = merge_patches(merged_cfg, cand.patch)
            cfg_path = cand_dir / "cfg_inference.yaml"
            write_yaml(cfg_path, merged_cfg)

            # 3) `results_dir` is managed by DMEval; planner may create a seed subdirectory inside.
            results_dir = cand_dir / "results"
            ensure_dir(results_dir)

            # 4) Serial runs: candidate × seed
            for seed in seeds:
                try:
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
                except Exception as exc:
                    _log(f"FAILED candidate={cand.candidate_id} seed={seed}: {exc}")
                    raise

            # 5) Extract artifacts: prefer JSONL, else fall back to .pt (adapter decides)
            run_tag = f"{sampler_name}__{cand.candidate_id}"
            rows = adapter.collect_trial_rows(
                results_root=results_dir,
                scenario=scenario_name,
                run_tag=run_tag,
                sampler=sampler_name,
                candidate_id=cand.candidate_id,
            )
            trial_rows.extend(rows)
            _log(
                f"{cand_idx:>4}/{len(candidates)} {cand.candidate_id} done "
                f"trials={len(rows)} elapsed={time.monotonic() - started_cand:.1f}s"
            )

        # Aggregation levels (aligned with the spec/paper terms):
        # - trial_df: one row per trial
        # - run_df: aggregated by seed (candidate × seed)
        # - candidate_df: aggregated across seeds (used by objective to choose best)
        trial_df = to_dataframe(trial_rows)
        run_df = aggregate_mean_std(df=trial_df, group_cols=["scenario", "sampler", "candidate_id", "run_tag", "seed"])
        candidate_df = aggregate_mean_std(df=trial_df, group_cols=["scenario", "sampler", "candidate_id"])

        write_stage_metrics(
            out_dir=sampler_dir,
            trial_df=trial_df,
            run_df=run_df,
            extra={"candidate_metrics": candidate_df},
        )

        # Objective selects top-k and best
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

            # `best_patch.yaml` is the core Stage-II input; `best.yaml` is for inspection (full merged cfg).
            write_yaml(sampler_best_dir / "best_patch.yaml", best_patch)
            best_full = merge_patches(read_yaml(base_cfg_path), best_patch)
            write_yaml(sampler_best_dir / "best.yaml", best_full)
            _log(
                f"best sampler={sampler_name} candidate={best_candidate_id} "
                f"success_mean={_fmt_num(best_row.get('success_mean'))} "
                f"time_mean={_fmt_num(best_row.get('t_inference_total_mean'))}"
            )
        else:
            _log(f"sampler={sampler_name} no best candidate (empty ranking)")

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
        _log(f"sampler={sampler_name} completed in {time.monotonic() - started_sampler:.1f}s")

    # Manifest records full config/search space/thresholds for reproducibility and traceability.
    write_yaml(pipeline_root / "tune_manifest.yaml", manifest)
    write_text(pipeline_root / "DONE_TUNE.txt", "OK\n")
    _section("STAGE I END")
    _log(f"completed in {time.monotonic() - started_all:.1f}s out={pipeline_root}")


def _build_adapter(cfg: Any) -> MPDAdapter:
    """
    Build an adapter.

    In this L1 build we only implement `MPDAdapter` (the first concrete adapter).
    When adding other planners, extend the dispatch logic here.
    """
    adapter_type = str(getattr(cfg.adapter, "type", "mpd")).lower()
    if adapter_type != "mpd":
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    return MPDAdapter.from_cfg(cfg.adapter)


def _to_plain(value: Any) -> Any:
    """Convert an OmegaConf container to plain Python dict/list (for manifest writing)."""
    try:
        # OmegaConf containers
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value
