from __future__ import annotations

import copy
import csv
import itertools
import os
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from dmeval.collect import collect_results
from dmeval.config import DMEvalConfig, JobConfig
from dmeval.hydra_loader import load_config
from dmeval.runner import run_jobs_serial
from dmeval.utils import (
    format_value_for_tag,
    is_nan,
    nanmean,
    nested_from_dotted_map,
    safe_token,
    set_by_dotted_key,
    to_float,
)


def _load_callable(spec: str) -> Callable[[Dict[str, Any]], Any]:
    spec = str(spec).strip()
    if ":" not in spec:
        raise ValueError(f"Invalid callable spec: {spec} (expected module:function)")
    module_name, fn_name = spec.split(":", 1)
    mod = import_module(module_name)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise ValueError(f"Callable not found: {spec}")
    return fn  # type: ignore[return-value]


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _ensure_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return [x]


def _grid_combinations(grid: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values_lists = [_ensure_list(grid[k]) for k in keys]
    for values in itertools.product(*values_lists):
        yield dict(zip(keys, values))


def _build_run_tag(sampler: str, patch: Dict[str, Any], *, max_len: int = 160) -> str:
    items: List[str] = []
    for k in sorted(patch.keys()):
        short_k = k.split(".")[-1]
        items.append(f"{safe_token(short_k)}={format_value_for_tag(patch[k])}")
    tag = safe_token(sampler)
    if items:
        tag = tag + "__" + "__".join(items)
    if len(tag) <= max_len:
        return tag

    # Too long: keep a stable hash suffix.
    import hashlib

    h = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:10]
    prefix = tag[: max(8, max_len - 12)].rstrip("_")
    return f"{prefix}__{h}"


def _rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class TuneCandidate:
    sampler: str
    method: str
    run_tag: str
    patch: Dict[str, Any]  # dotted-key -> value
    cfg_path: str


def _extract_sampling_patch(candidate: TuneCandidate) -> Dict[str, Any]:
    """
    Convert dotted-key patch + diffusion_sampling_method into a nested patch dict.
    This patch is meant to be applied on top of a scenario base config.
    """
    nested = nested_from_dotted_map(candidate.patch)
    nested["diffusion_sampling_method"] = candidate.method
    return nested


def _select_best(
    run_rows: List[Dict[str, Any]],
    candidates_by_tag: Dict[str, TuneCandidate],
    *,
    score_fn: Callable[[Dict[str, Any]], float],
    feasible_fn: Callable[[Dict[str, Any]], bool],
    top_k: int,
) -> Tuple[Dict[str, TuneCandidate], Dict[str, List[Tuple[TuneCandidate, float]]]]:
    # Score each (run_tag, seed) row, then aggregate across seeds per run_tag.
    scores_by_tag: Dict[str, List[float]] = {}
    feasible_by_tag: Dict[str, List[bool]] = {}
    row_by_tag: Dict[str, List[Dict[str, Any]]] = {}

    for r in run_rows:
        run_tag = str(r.get("run_tag", ""))
        if run_tag not in candidates_by_tag:
            continue
        ok = bool(feasible_fn(r))
        s = float(score_fn(r))
        feasible_by_tag.setdefault(run_tag, []).append(ok)
        scores_by_tag.setdefault(run_tag, []).append(s)
        row_by_tag.setdefault(run_tag, []).append(r)

    # Aggregate per run_tag (mean across seeds).
    agg_score: Dict[str, float] = {}
    agg_feasible: Dict[str, bool] = {}
    for run_tag, scores in scores_by_tag.items():
        agg_score[run_tag] = nanmean(scores)
        agg_feasible[run_tag] = all(feasible_by_tag.get(run_tag, [False]))

    # Group by sampler.
    ranked: Dict[str, List[Tuple[TuneCandidate, float]]] = {}
    for run_tag, cand in candidates_by_tag.items():
        if run_tag not in agg_score:
            continue
        if not agg_feasible.get(run_tag, False):
            continue
        ranked.setdefault(cand.sampler, []).append((cand, agg_score[run_tag]))

    # Sort and keep top_k.
    for sampler, items in ranked.items():
        ranked[sampler] = sorted(items, key=lambda x: x[1], reverse=True)[: max(1, top_k)]

    best: Dict[str, TuneCandidate] = {}
    for sampler, items in ranked.items():
        if items:
            best[sampler] = items[0][0]

    return best, ranked


def tune_command(*, config_path: str, overrides: Optional[List[str]] = None) -> int:
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

    tune_cfg = cfg.get("tune", {}) or {}
    base_cfg_path = str(tune_cfg.get("base_cfg", ""))
    if not base_cfg_path:
        print("[dmeval] ERROR: tune.base_cfg is required.")
        return 1

    scenario_name = safe_token(str(tune_cfg.get("scenario_name") or Path(base_cfg_path).stem))
    stage1_root = Path(pipeline_root) / "stage1_tune" / scenario_name

    seeds = [int(s) for s in (tune_cfg.get("seeds", [2]) or [2])]
    samplers = tune_cfg.get("samplers", [])
    if not isinstance(samplers, list) or not samplers:
        print("[dmeval] ERROR: tune.samplers must be a non-empty list.")
        return 1

    obj_cfg = dict(tune_cfg.get("objective", {}) or {})
    score_fn = _load_callable(obj_cfg.get("score_fn", "dmeval.objectives.simple:score"))
    feasible_fn = _load_callable(obj_cfg.get("feasible_fn", "dmeval.objectives.simple:is_feasible"))
    top_k = int(tune_cfg.get("top_k", 1) or 1)

    dry_run = bool(tune_cfg.get("dry_run", False))
    skip_existing = bool(tune_cfg.get("skip_existing", True))

    # Resolve stage1 root to absolute path (relative to workdir).
    base_dir = Path(workdir).resolve()
    stage1_root_abs = (base_dir / stage1_root) if not stage1_root.is_absolute() else stage1_root
    stage1_root_abs.mkdir(parents=True, exist_ok=True)

    dmeval_dir = stage1_root_abs / "_dmeval"
    gen_cfg_dir = dmeval_dir / "generated_configs"
    best_dir = dmeval_dir / "best_configs"
    gen_cfg_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(Path(base_cfg_path))

    candidates: List[TuneCandidate] = []
    for s in samplers:
        if not isinstance(s, dict):
            continue
        sampler_name = str(s.get("name", "")).strip()
        if not sampler_name:
            continue
        method = str(s.get("method", sampler_name)).strip()
        grid = dict(s.get("grid", {}) or {})

        for patch in _grid_combinations(grid):
            run_tag = _build_run_tag(sampler_name, patch)

            cfg_out = copy.deepcopy(base_cfg)
            cfg_out["diffusion_sampling_method"] = method
            for k, v in patch.items():
                set_by_dotted_key(cfg_out, k, v)

            out_path = gen_cfg_dir / sampler_name / f"{run_tag}.yaml"
            _dump_yaml(cfg_out, out_path)
            candidates.append(
                TuneCandidate(
                    sampler=sampler_name,
                    method=method,
                    run_tag=run_tag,
                    patch=patch,
                    cfg_path=str(out_path),
                )
            )

    if not candidates:
        print("[dmeval] ERROR: no tuning candidates generated.")
        return 1

    manifest_path = dmeval_dir / "tune_manifest.yaml"
    _dump_yaml(
        {
            "scenario_name": scenario_name,
            "base_cfg": base_cfg_path,
            "stage1_root": str(stage1_root_abs),
            "candidates": [
                {
                    "sampler": c.sampler,
                    "method": c.method,
                    "run_tag": c.run_tag,
                    "cfg_path": c.cfg_path,
                    "patch": c.patch,
                }
                for c in candidates
            ],
        },
        manifest_path,
    )

    # Build jobs and run (serial).
    jobs: List[JobConfig] = []
    for c in candidates:
        run_tag_path = c.run_tag  # stage1 layout: <stage1_root>/<run_tag>/<seed>/*
        results_dir = "{results_root}/{run_tag}"
        cmd = [
            python_bin,
            mpd_infer_script,
            "--cfg_inference_path",
            c.cfg_path,
            "--results_dir",
            results_dir,
            "--seed",
            "{seed}",
        ]
        # Append common args (converted to CLI tokens).
        for k, v in common_args.items():
            cmd.append(f"--{k}")
            cmd.append(str(v).lower() if isinstance(v, bool) else str(v))

        jobs.append(JobConfig(run_tag=run_tag_path, cmd=cmd, seeds=seeds, workdir=workdir))

    # Optionally skip runs if results exist.
    if skip_existing and not dry_run:
        filtered: List[JobConfig] = []
        for j in jobs:
            missing_any = False
            for seed in j.seeds:
                seed_dir = stage1_root_abs / j.run_tag / str(seed)
                if not any(seed_dir.glob("results_single_plan-*.pt")):
                    missing_any = True
                    break
            if missing_any:
                filtered.append(j)
        jobs = filtered

    if jobs:
        run_jobs_serial(DMEvalConfig(version=1, results_root=str(stage1_root_abs), defaults={"workdir": workdir}, jobs=jobs), dry_run=dry_run)
    else:
        print("[dmeval] All candidates already have results; skip execution.")

    # Collect (aggregate then analyze).
    try:
        collect_results(adapter="mpd", results_root=str(stage1_root_abs), out_dir=str(dmeval_dir), include_all_config=False)
    except Exception as exc:
        print(f"[dmeval] ERROR during collect: {exc}")
        return 1

    run_metrics_path = dmeval_dir / "run_metrics.csv"
    if not run_metrics_path.exists():
        print(f"[dmeval] ERROR: missing run_metrics.csv at {run_metrics_path}")
        return 1

    run_rows = _rows_from_csv(run_metrics_path)

    candidates_by_tag = {c.run_tag: c for c in candidates}
    best, ranked = _select_best(
        run_rows,
        candidates_by_tag,
        score_fn=score_fn,
        feasible_fn=feasible_fn,
        top_k=top_k,
    )

    if not best:
        print("[dmeval] ERROR: no feasible candidates found (check feasibility/objective).")
        return 1

    # Write best configs into a dedicated folder (per sampler).
    summary: Dict[str, Any] = {"scenario_name": scenario_name, "best": {}}
    for sampler, cand in best.items():
        dst_dir = best_dir / sampler
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy the full YAML used in MPD inference.
        shutil.copy2(cand.cfg_path, dst_dir / "best.yaml")

        # Also store a "patch" YAML (sampler hyperparams only) for Stage II reuse.
        patch = _extract_sampling_patch(cand)
        _dump_yaml(patch, dst_dir / "best_patch.yaml")

        # Store top-k info.
        topk_items = ranked.get(sampler, [])
        _dump_yaml(
            [{"run_tag": c.run_tag, "cfg_path": c.cfg_path, "score": float(score)} for c, score in topk_items],
            dst_dir / "topk.yaml",
        )

        summary["best"][sampler] = {
            "run_tag": cand.run_tag,
            "cfg_path": cand.cfg_path,
            "method": cand.method,
            "best_yaml": str(dst_dir / "best.yaml"),
            "best_patch_yaml": str(dst_dir / "best_patch.yaml"),
        }

    _dump_yaml(summary, best_dir / "summary.yaml")
    print(f"[dmeval] Wrote best configs to {best_dir}")
    return 0
