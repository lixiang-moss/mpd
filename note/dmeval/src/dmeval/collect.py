from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dmeval.adapters.mpd import MpdResultsAdapter
from dmeval.utils import is_nan, nanmean, nanstd, to_float


def _get_adapter(name: str):
    name = (name or "").strip().lower()
    if name == "mpd":
        return MpdResultsAdapter()
    raise ValueError(f"Unknown adapter: {name}")


def _write_csv(path: Path, rows: List[Dict[str, Any]], *, fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_keys(rows: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                keys.add(k)
    return sorted(keys)


def _group_trials(
    rows: List[Dict[str, Any]],
    *,
    group_keys: Tuple[str, ...],
    numeric_keys: List[str],
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[tuple(r.get(k, "") for k in group_keys)].append(r)

    out: List[Dict[str, Any]] = []
    for group, group_rows in buckets.items():
        row: Dict[str, Any] = {k: v for k, v in zip(group_keys, group)}
        row["n_trials"] = len(group_rows)
        for nk in numeric_keys:
            values = [to_float(gr.get(nk, float("nan"))) for gr in group_rows]
            row[f"{nk}_mean"] = nanmean(values)
            row[f"{nk}_std"] = nanstd(values)
        out.append(row)
    return out


def _rank_rows(rows: List[Dict[str, Any]], *, key_name: str, descending: bool) -> List[Dict[str, Any]]:
    def _key(r: Dict[str, Any]) -> float:
        v = to_float(r.get(key_name, float("nan")))
        if is_nan(v):
            return float("-inf") if descending else float("inf")
        return v

    return sorted(rows, key=_key, reverse=descending)


def collect_results(
    *,
    adapter: str,
    results_root: str,
    out_dir: str,
    include_all_config: bool,
) -> None:
    results_root_p = Path(results_root)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    adapter_obj = _get_adapter(adapter)

    trial_rows: List[Dict[str, Any]] = []
    n_skipped = 0
    for artifact in adapter_obj.discover_trials(results_root_p):
        try:
            trial_obj = adapter_obj.load_trial_object(artifact)
            row = adapter_obj.extract_row(artifact, trial_obj, include_all_config=include_all_config)
            trial_rows.append(row)
        except Exception as exc:
            n_skipped += 1
            print(f"[dmeval] Skip {artifact.result_path}: {exc}")

    if not trial_rows:
        raise RuntimeError("No results found.")

    # Keep a stable header order (core fields first, then the rest).
    core = [
        "scenario",
        "env_id_replace",
        "run_tag",
        "seed",
        "result_file",
        "success",
        "fraction_valid",
        "collision_intensity",
        "path_length_best",
        "smoothness_best",
        "diversity_valid",
        "t_inference_total",
        "t_generator",
        "t_guide",
        "diffusion_sampling_method",
        "planner_alg",
    ]
    all_keys = {k for r in trial_rows for k in r.keys()}
    extra_keys = sorted(all_keys - set(core))
    field_order = [k for k in core if k in all_keys] + extra_keys
    trial_rows = [{k: r.get(k, "") for k in field_order} for r in trial_rows]

    trial_csv = out_dir_p / "trial_metrics.csv"
    _write_csv(trial_csv, trial_rows, fieldnames=field_order)
    print(f"[dmeval] Wrote {trial_csv} (skipped={n_skipped})")

    numeric_keys = _numeric_keys(trial_rows)
    run_rows = _group_trials(trial_rows, group_keys=("scenario", "run_tag", "seed"), numeric_keys=numeric_keys)
    run_csv = out_dir_p / "run_metrics.csv"
    _write_csv(run_csv, run_rows)
    print(f"[dmeval] Wrote {run_csv}")

    # Rankings based on run-level means (more meaningful than per-trial ranking).
    def _write_rank(name: str, ranked: List[Dict[str, Any]]) -> None:
        path = out_dir_p / name
        _write_csv(path, ranked)
        print(f"[dmeval] Wrote {path}")

    _write_rank("rank_success.csv", _rank_rows(run_rows, key_name="success_mean", descending=True))
    _write_rank("rank_fraction_valid.csv", _rank_rows(run_rows, key_name="fraction_valid_mean", descending=True))
    _write_rank("rank_path_length.csv", _rank_rows(run_rows, key_name="path_length_best_mean", descending=False))
    _write_rank("rank_speed.csv", _rank_rows(run_rows, key_name="t_inference_total_mean", descending=False))


def collect_command(*, adapter: str, results_root: str, out_dir: str, include_all_config: bool) -> int:
    try:
        collect_results(
            adapter=adapter,
            results_root=results_root,
            out_dir=out_dir,
            include_all_config=include_all_config,
        )
    except Exception as exc:
        print(f"[dmeval] ERROR: {exc}")
        return 1
    return 0
