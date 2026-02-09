"""
Stage II: compare (fair comparison stage).

Responsibilities (per spec):
- Read Stage-I outputs `best_configs/<sampler>/best_patch.yaml`
- Apply patches to each scenario base_cfg to generate `cfg_inference.yaml`
- Run the system-under-test serially: scenario × sampler × seed
- Collect: extract trial rows → aggregate to run/agg metrics
- Rank + Plot: write ranking CSVs and basic plots

Key constraints:
- Serial only (no parallel scheduling)
- No metric recomputation; extraction and aggregation only
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

import shutil

from .adapters import MPDAdapter
from .collect import aggregate_mean_std, to_dataframe, write_stage_metrics, write_csv
from .plotting import plot_stage_compare
from .runner import run_planner_subprocess
from .util import cfg_to_yaml_str, ensure_dir, merge_patches, read_yaml, write_text, write_yaml

_LINE = "-" * 72


def _log(msg: str) -> None:
    """Stage-II progress logging."""
    print(f"[dmeval:compare] {msg}", flush=True)


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


def run_compare(cfg: Any) -> None:
    """Run Stage II (compare)."""
    if not bool(getattr(cfg.compare, "enabled", True)):
        _log("skip (compare.enabled=false)")
        return

    started_all = time.monotonic()
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
        # `compare/` is a derived artifact directory; allow rebuilding when `allow_overwrite=true`.
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
    _section("STAGE II START")
    _log(
        f"start pipeline_root={pipeline_root} scenarios={len(scenarios)} "
        f"samplers={len(sampler_names)} seeds={seeds}"
    )

    # `trial_rows` is the unified "one row per trial" record format (from the adapter).
    trial_rows: list[dict[str, Any]] = []

    for scenario_idx, scenario_cfg in enumerate(scenarios, start=1):
        scenario_name = str(scenario_cfg.name)
        base_cfg_path = Path(str(scenario_cfg.base_cfg))
        _section(f"SCENARIO {scenario_idx}/{len(scenarios)} | {scenario_name}")

        for sampler_idx, sampler_name in enumerate(sampler_names, start=1):
            started_pair = time.monotonic()
            # Core Stage-II input: Stage-I persisted `best_patch.yaml`
            patch_path = best_root / sampler_name / "best_patch.yaml"
            if not patch_path.exists():
                _log(f"  sampler {sampler_idx}/{len(sampler_names)} name={sampler_name} skip (missing best_patch)")
                continue
            patch = read_yaml(patch_path)
            # Build final inference config: scenario base_cfg + sampler best_patch
            merged_cfg = merge_patches(read_yaml(base_cfg_path), patch)

            run_dir = stage_dir / scenario_name / sampler_name
            ensure_dir(run_dir)
            cfg_path = run_dir / "cfg_inference.yaml"
            write_yaml(cfg_path, merged_cfg)

            results_dir = run_dir / "results"
            ensure_dir(results_dir)

            # Serial runs: scenario × sampler × seed
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
                        log_path=run_dir / f"seed{seed}.log",
                    )
                except Exception as exc:
                    _log(f"FAILED scenario={scenario_name} sampler={sampler_name} seed={seed}: {exc}")
                    raise

            # Extract artifacts: adapter decides whether to read JSONL or .pt
            run_tag = sampler_name
            rows = adapter.collect_trial_rows(
                results_root=results_dir,
                scenario=scenario_name,
                run_tag=run_tag,
                sampler=sampler_name,
                candidate_id=None,
            )
            trial_rows.extend(rows)
            _log(
                f"{sampler_idx:>3}/{len(sampler_names)} sampler={sampler_name} done trials={len(rows)} "
                f"elapsed={time.monotonic() - started_pair:.1f}s"
            )

    # Aggregation levels (aligned with spec/paper terms):
    # - run_df: aggregated by seed (scenario × sampler × seed)
    # - run_agg_df: aggregated across seeds (scenario × sampler) for fair comparison/plotting/ranking
    trial_df = to_dataframe(trial_rows)
    run_df = aggregate_mean_std(df=trial_df, group_cols=["scenario", "sampler", "run_tag", "seed"])
    run_agg_df = aggregate_mean_std(df=trial_df, group_cols=["scenario", "sampler", "run_tag"])

    write_stage_metrics(out_dir=stage_dir, trial_df=trial_df, run_df=run_df, extra={"run_metrics_agg": run_agg_df})

    _write_rankings(run_agg_df, out_dir=stage_dir)
    plot_stage_compare(run_metrics_agg=run_agg_df, out_dir=stage_dir / "plots")
    _log(
        f"aggregated rows: trial={len(trial_df)} run={len(run_df)} run_agg={len(run_agg_df)} "
        f"metrics={stage_dir / 'run_metrics.csv'}"
    )
    if not run_agg_df.empty:
        rank_df = run_agg_df.copy()
        if "success_mean" not in rank_df.columns:
            rank_df["success_mean"] = float("-inf")
        if "t_inference_total_mean" not in rank_df.columns:
            rank_df["t_inference_total_mean"] = float("inf")
        best_row = rank_df.sort_values(["success_mean", "t_inference_total_mean"], ascending=[False, True]).iloc[0]
        _log(
            "best summary: "
            f"scenario={best_row.get('scenario')} sampler={best_row.get('sampler')} "
            f"success_mean={_fmt_num(best_row.get('success_mean'))} "
            f"time_mean={_fmt_num(best_row.get('t_inference_total_mean'))}"
        )

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
    _section("STAGE II END")
    _log(f"completed in {time.monotonic() - started_all:.1f}s out={stage_dir}")


def _build_adapter(cfg: Any) -> MPDAdapter:
    """Build an adapter (this L1 build implements MPDAdapter only)."""
    adapter_type = str(getattr(cfg.adapter, "type", "mpd")).lower()
    if adapter_type != "mpd":
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
    return MPDAdapter.from_cfg(cfg.adapter)


def _discover_samplers(best_root: Path) -> list[str]:
    """Discover sampler names under `best_configs_root` (directory must contain `best_patch.yaml`)."""
    out: list[str] = []
    if not best_root.exists():
        return out
    for p in best_root.iterdir():
        if p.is_dir() and (p / "best_patch.yaml").exists():
            out.append(p.name)
    return sorted(out)


def _write_rankings(run_agg_df: pd.DataFrame, *, out_dir: Path) -> None:
    """
    Write basic ranking CSVs (minimum required by the spec).

    Ranking rules (earlier is better):
    - success: `success_mean` desc, then `t_inference_total_mean` asc
    - fraction_valid: `fraction_valid_mean` desc, then `t_inference_total_mean` asc
    - path_length: `path_length_best_mean` asc
    - speed: `t_inference_total_mean` asc
    """
    if run_agg_df.empty:
        return

    def _safe(col: str, default: float) -> pd.Series:
        # Robustness: use `default` when column is missing or non-numeric.
        if col not in run_agg_df.columns:
            return pd.Series([default] * len(run_agg_df))
        s = pd.to_numeric(run_agg_df[col], errors="coerce").fillna(default)
        return s

    # Build `_k1`/`_k2` as ranking keys; smaller is better.
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
