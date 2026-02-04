from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from dmeval.collect import collect_results
from dmeval.config import DMEvalConfig, JobConfig
from dmeval.hydra_loader import load_config
from dmeval.plot import plot_from_run_metrics
from dmeval.runner import run_jobs_serial
from dmeval.utils import safe_token


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k, None), dict):
            out[k] = _deep_merge(out[k], v)
            continue
        out[k] = copy.deepcopy(v)
    return out


def compare_command(*, config_path: str, overrides: Optional[List[str]] = None) -> int:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"[dmeval] ERROR: config not found: {cfg_path}")
        return 1

    try:
        cfg = load_config(config_path, overrides=overrides)
    except Exception as exc:
        print(f"[dmeval] ERROR: failed to load config via Hydra: {exc}")
        return 1

    workdir = str(cfg.get("workdir", "."))
    pipeline_root = str(cfg.get("pipeline_root", "mpd-splines-public/scripts/inference/logs/dmeval_pipeline"))
    python_bin = str(cfg.get("python", "python3"))
    mpd_infer_script = str(cfg.get("mpd_inference_script", "mpd-splines-public/scripts/inference/inference.py"))
    common_args = dict(cfg.get("common_inference_args", {}) or {})

    compare_cfg = cfg.get("compare", {}) or {}
    best_configs_root = Path(str(compare_cfg.get("best_configs_root", "")))
    if not best_configs_root.exists():
        print("[dmeval] ERROR: compare.best_configs_root is required and must exist.")
        return 1

    scenarios = compare_cfg.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        print("[dmeval] ERROR: compare.scenarios must be a non-empty list.")
        return 1

    seeds = [int(s) for s in (compare_cfg.get("seeds", [2]) or [2])]
    dry_run = bool(compare_cfg.get("dry_run", False))
    skip_existing = bool(compare_cfg.get("skip_existing", True))
    make_plots = bool(compare_cfg.get("make_plots", True))

    stage2_root = Path(pipeline_root) / "stage2_compare"
    base_dir = Path(workdir).resolve()
    stage2_root_abs = (base_dir / stage2_root) if not stage2_root.is_absolute() else stage2_root
    stage2_root_abs.mkdir(parents=True, exist_ok=True)

    dmeval_dir = stage2_root_abs / "_dmeval"
    gen_cfg_dir = dmeval_dir / "generated_configs"
    gen_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Load best patches (per sampler).
    sampler_dirs = [p for p in best_configs_root.iterdir() if p.is_dir()]
    if not sampler_dirs:
        print(f"[dmeval] ERROR: no sampler dirs found under {best_configs_root}")
        return 1

    sampler_patches: Dict[str, Dict[str, Any]] = {}
    for p in sampler_dirs:
        patch_path = p / "best_patch.yaml"
        if not patch_path.exists():
            # Fallback to full config, but this is less ideal for cross-scenario reuse.
            full_path = p / "best.yaml"
            if not full_path.exists():
                continue
            sampler_patches[p.name] = _load_yaml(full_path)
            continue
        sampler_patches[p.name] = _load_yaml(patch_path)

    if not sampler_patches:
        print(f"[dmeval] ERROR: no best_patch.yaml (or best.yaml) found under {best_configs_root}")
        return 1

    # Generate scenario-specific configs by applying patches to each scenario base cfg.
    jobs: List[JobConfig] = []
    for s in scenarios:
        if not isinstance(s, dict):
            continue
        scenario_name = safe_token(str(s.get("name", "")))
        base_cfg_path = str(s.get("base_cfg", ""))
        if not scenario_name or not base_cfg_path:
            continue

        base_cfg = _load_yaml(Path(base_cfg_path))

        for sampler, patch in sampler_patches.items():
            cfg_out = _deep_merge(base_cfg, patch)
            out_path = gen_cfg_dir / scenario_name / f"{sampler}.yaml"
            _dump_yaml(cfg_out, out_path)

            run_tag_path = f"{scenario_name}/{sampler}"  # stage2 layout: <stage2_root>/<scenario>/<sampler>/<seed>
            cmd = [
                python_bin,
                mpd_infer_script,
                "--cfg_inference_path",
                str(out_path),
                "--results_dir",
                "{results_root}/{run_tag}",
                "--seed",
                "{seed}",
            ]
            for k, v in common_args.items():
                cmd.append(f"--{k}")
                cmd.append(str(v).lower() if isinstance(v, bool) else str(v))

            jobs.append(JobConfig(run_tag=run_tag_path, cmd=cmd, seeds=seeds, workdir=workdir))

    if not jobs:
        print("[dmeval] ERROR: no compare jobs generated (check scenarios and best configs).")
        return 1

    # Optionally skip runs if results exist.
    if skip_existing and not dry_run:
        filtered: List[JobConfig] = []
        for j in jobs:
            missing_any = False
            for seed in j.seeds:
                seed_dir = stage2_root_abs / j.run_tag / str(seed)
                if not any(seed_dir.glob("results_single_plan-*.pt")):
                    missing_any = True
                    break
            if missing_any:
                filtered.append(j)
        jobs = filtered

    if jobs:
        run_jobs_serial(DMEvalConfig(version=1, results_root=str(stage2_root_abs), defaults={"workdir": workdir}, jobs=jobs), dry_run=dry_run)
    else:
        print("[dmeval] All compare jobs already have results; skip execution.")

    # Collect (aggregate then analyze).
    try:
        collect_results(adapter="mpd", results_root=str(stage2_root_abs), out_dir=str(dmeval_dir), include_all_config=False)
    except Exception as exc:
        print(f"[dmeval] ERROR during collect: {exc}")
        return 1

    if make_plots:
        try:
            plot_from_run_metrics(run_metrics=dmeval_dir / "run_metrics.csv", out_dir=dmeval_dir / "plots", scenario="")
        except Exception as exc:
            print(f"[dmeval] WARNING: plot failed: {exc}")

    print(f"[dmeval] Stage II results: {dmeval_dir}")
    return 0
