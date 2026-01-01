import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


_PW_RE = re.compile(r"_pw(?P<pw>[0-9p]+)")


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _nanmean(xs: Sequence[float]) -> float:
    vals = [v for v in xs if v == v]
    if not vals:
        return float("nan")
    return float(sum(vals) / float(len(vals)))


def _nanmin(xs: Sequence[float]) -> float:
    vals = [v for v in xs if v == v]
    if not vals:
        return float("nan")
    return float(min(vals))


def _nanmax(xs: Sequence[float]) -> float:
    vals = [v for v in xs if v == v]
    if not vals:
        return float("nan")
    return float(max(vals))


def _parse_pw(run_tag: str) -> Optional[float]:
    m = _PW_RE.search(run_tag)
    if not m:
        return None
    token = m.group("pw")
    if not token:
        return None
    try:
        return float(token.replace("p", "."))
    except Exception:
        return None


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@dataclass
class PerSeedRow:
    sweep_label: str
    env_label: str
    run_tag: str
    seed: int
    pw: float
    # primary
    success: float
    fraction_valid: float
    # extras (optional)
    collision_intensity: float
    smoothness_best: float
    path_length_best: float
    t_inference_total: float
    diversity_valid: float
    n_trajectory_samples: float


def _load_results_diversity(result_path: str) -> float:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing torch; run inside the project conda env (the same env used for inference)."
        ) from exc

    result = torch.load(result_path, map_location="cpu")
    metrics = getattr(result, "metrics", None)
    trajs_valid = getattr(metrics, "trajs_valid", None)
    div = getattr(trajs_valid, "diversity", None)
    if div is None:
        return float("nan")
    try:
        return float(div)
    except Exception:
        return float("nan")


def _read_sweep_metrics(results_root: str) -> List[Dict[str, str]]:
    csv_path = os.path.join(results_root, "sweep_metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _filter_rows_by_run_tag_regex(rows: Sequence[Dict[str, str]], run_tag_regex: str) -> List[Dict[str, str]]:
    if not run_tag_regex:
        return list(rows)
    rx = re.compile(run_tag_regex)
    out = []
    for r in rows:
        tag = (r.get("run_tag") or "").strip()
        if tag and rx.search(tag):
            out.append(r)
    return out


def _collect_per_seed_rows(
    *,
    results_root: str,
    sweep_label: str,
    env_label: str,
    run_tag_regex: str,
    pw_min: float,
    pw_max: float,
) -> List[PerSeedRow]:
    rows = _read_sweep_metrics(results_root)
    rows = _filter_rows_by_run_tag_regex(rows, run_tag_regex=run_tag_regex)

    out: List[PerSeedRow] = []
    for r in rows:
        run_tag = (r.get("run_tag") or "").strip()
        seed_raw = (r.get("seed") or "").strip()
        result_file = (r.get("result_file") or "").strip()
        pw = _parse_pw(run_tag)
        if pw is None:
            continue
        if pw < pw_min or pw > pw_max:
            continue

        try:
            seed = int(seed_raw)
        except Exception:
            continue

        run_dir = os.path.join(results_root, run_tag, str(seed))
        result_path = os.path.join(run_dir, result_file)

        args_inference = _read_yaml(os.path.join(run_dir, "args_inference.yaml"))
        n_trajectory_samples = _to_float(args_inference.get("n_trajectory_samples"))

        diversity_valid = float("nan")
        if os.path.exists(result_path):
            try:
                diversity_valid = _load_results_diversity(result_path)
            except Exception:
                diversity_valid = float("nan")

        out.append(
            PerSeedRow(
                sweep_label=sweep_label,
                env_label=env_label,
                run_tag=run_tag,
                seed=seed,
                pw=pw,
                success=_to_float(r.get("success")),
                fraction_valid=_to_float(r.get("fraction_valid")),
                collision_intensity=_to_float(r.get("collision_intensity")),
                smoothness_best=_to_float(r.get("smoothness_best")),
                path_length_best=_to_float(r.get("path_length_best")),
                t_inference_total=_to_float(r.get("t_inference_total")),
                diversity_valid=diversity_valid,
                n_trajectory_samples=n_trajectory_samples,
            )
        )
    return out


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_by_pw(per_seed: Sequence[PerSeedRow]) -> List[Dict[str, Any]]:
    by: Dict[Tuple[str, str, float], List[PerSeedRow]] = {}
    for r in per_seed:
        by.setdefault((r.env_label, r.sweep_label, r.pw), []).append(r)

    rows: List[Dict[str, Any]] = []
    for (env_label, sweep_label, pw), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        fv = [r.fraction_valid for r in rs]
        succ = [r.success for r in rs]
        div = [r.diversity_valid for r in rs]
        coll = [r.collision_intensity for r in rs]
        smooth = [r.smoothness_best for r in rs]
        path = [r.path_length_best for r in rs]
        t_total = [r.t_inference_total for r in rs]
        n_traj = [r.n_trajectory_samples for r in rs]

        rows.append(
            {
                "env_label": env_label,
                "sweep_label": sweep_label,
                "pw": pw,
                "n_rows": len(rs),
                "success_mean": _nanmean(succ),
                "success_min": _nanmin(succ),
                "fraction_valid_mean": _nanmean(fv),
                "fraction_valid_min": _nanmin(fv),
                "fraction_valid_max": _nanmax(fv),
                "diversity_valid_mean": _nanmean(div),
                "diversity_valid_min": _nanmin(div),
                "diversity_valid_max": _nanmax(div),
                "collision_intensity_mean": _nanmean(coll),
                "smoothness_best_mean": _nanmean(smooth),
                "path_length_best_mean": _nanmean(path),
                "t_inference_total_mean": _nanmean(t_total),
                "n_trajectory_samples_mean": _nanmean(n_traj),
            }
        )
    return rows


def _plot_line(
    out_path: str,
    series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    plt.figure(figsize=(10, 6))
    for label, (xs, ys) in series.items():
        xs_arr = np.asarray(list(xs), dtype=np.float64)
        ys_arr = np.asarray(list(ys), dtype=np.float64)
        mask = np.isfinite(xs_arr) & np.isfinite(ys_arr)
        xs_arr = xs_arr[mask]
        ys_arr = ys_arr[mask]
        if xs_arr.size == 0:
            continue
        order = np.argsort(xs_arr)
        xs_arr = xs_arr[order]
        ys_arr = ys_arr[order]
        plt.plot(xs_arr, ys_arr, marker="o", linewidth=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="mpd-splines-public/scripts/inference/logs/compare_prior_weight",
        help="Output directory for comparison tables and plots.",
    )
    parser.add_argument(
        "--sweep_a_pw_to2_steps44",
        default="mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_pw_to2_steps44",
        help="Env A sweep root (pw 1.1..2.0, steps=44 baseline).",
    )
    parser.add_argument(
        "--sweep_replace_pw_to2_steps44",
        default="mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_pw_to2_steps44_replace",
        help="Env replace sweep root (pw 0.95..2.0, steps=44 baseline).",
    )
    parser.add_argument(
        "--sweep_a_pw_v1",
        default="mpd-splines-public/scripts/inference/logs/sweep_dpm_solver_pw_v1",
        help="Env A sweep root with low pw coverage (used only after filtering run_tag).",
    )
    parser.add_argument(
        "--a_pw_v1_run_tag_regex",
        default=r"dpm_solver_pp_steps44_order2_ct1_mmultistep_skiplogSNR_soldpmsolver_lof0_dz1_pw[0-9p]+_ng1_glr0p012_mpx0p1_tsg0p1$",
        help="Regex filter for extracting an apples-to-apples steps44 baseline slice from sweep_a_pw_v1.",
    )
    parser.add_argument("--pw_min", type=float, default=0.0)
    parser.add_argument("--pw_max", type=float, default=10.0)
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Collect per-seed rows.
    per_seed: List[PerSeedRow] = []
    per_seed += _collect_per_seed_rows(
        results_root=args.sweep_a_pw_to2_steps44,
        sweep_label="A_pw_to2_steps44",
        env_label="EnvA",
        run_tag_regex=r"_pw[0-9p]+",
        pw_min=args.pw_min,
        pw_max=args.pw_max,
    )
    per_seed += _collect_per_seed_rows(
        results_root=args.sweep_replace_pw_to2_steps44,
        sweep_label="Replace_pw_to2_steps44",
        env_label="EnvReplace",
        run_tag_regex=r"_pw[0-9p]+",
        pw_min=args.pw_min,
        pw_max=args.pw_max,
    )
    per_seed += _collect_per_seed_rows(
        results_root=args.sweep_a_pw_v1,
        sweep_label="A_pw_v1_steps44_baseline",
        env_label="EnvA",
        run_tag_regex=args.a_pw_v1_run_tag_regex,
        pw_min=args.pw_min,
        pw_max=args.pw_max,
    )

    per_seed_csv = os.path.join(out_dir, "per_seed_metrics_with_diversity.csv")
    _write_csv(
        per_seed_csv,
        [
            {
                "env_label": r.env_label,
                "sweep_label": r.sweep_label,
                "run_tag": r.run_tag,
                "seed": r.seed,
                "pw": r.pw,
                "success": r.success,
                "fraction_valid": r.fraction_valid,
                "diversity_valid": r.diversity_valid,
                "collision_intensity": r.collision_intensity,
                "smoothness_best": r.smoothness_best,
                "path_length_best": r.path_length_best,
                "t_inference_total": r.t_inference_total,
                "n_trajectory_samples": r.n_trajectory_samples,
            }
            for r in per_seed
        ],
    )

    agg = _aggregate_by_pw(per_seed)
    agg_csv = os.path.join(out_dir, "aggregated_by_pw.csv")
    _write_csv(agg_csv, agg)

    # Build plotting series (pw -> metric) per env_label (and optionally per sweep_label).
    def _series(metric_key: str, env_label: str, sweep_label: Optional[str] = None) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for r in agg:
            if r["env_label"] != env_label:
                continue
            if sweep_label is not None and r["sweep_label"] != sweep_label:
                continue
            xs.append(float(r["pw"]))
            ys.append(float(r.get(metric_key, float("nan"))))
        return xs, ys

    # Main comparisons: EnvA vs EnvReplace using the isolated sweeps.
    series_fv_min = {
        "EnvA (steps44 baseline)": _series("fraction_valid_min", "EnvA", sweep_label="A_pw_to2_steps44"),
        "EnvReplace (steps44 baseline)": _series("fraction_valid_min", "EnvReplace"),
    }
    _plot_line(
        os.path.join(out_dir, "pw_vs_fraction_valid_min.png"),
        series=series_fv_min,
        xlabel="prior_weight_with_guide (pw)",
        ylabel="fraction_valid_min (across seeds)",
        title="pw effect on robustness: fraction_valid_min",
        ylim=(0.0, 1.0),
    )

    series_fv_mean = {
        "EnvA (steps44 baseline)": _series("fraction_valid_mean", "EnvA", sweep_label="A_pw_to2_steps44"),
        "EnvReplace (steps44 baseline)": _series("fraction_valid_mean", "EnvReplace"),
    }
    _plot_line(
        os.path.join(out_dir, "pw_vs_fraction_valid_mean.png"),
        series=series_fv_mean,
        xlabel="prior_weight_with_guide (pw)",
        ylabel="fraction_valid_mean (across seeds)",
        title="pw effect on average feasibility: fraction_valid_mean",
        ylim=(0.0, 1.0),
    )

    series_success = {
        "EnvA (steps44 baseline)": _series("success_mean", "EnvA", sweep_label="A_pw_to2_steps44"),
        "EnvReplace (steps44 baseline)": _series("success_mean", "EnvReplace"),
    }
    _plot_line(
        os.path.join(out_dir, "pw_vs_success_mean.png"),
        series=series_success,
        xlabel="prior_weight_with_guide (pw)",
        ylabel="success_mean (across seeds)",
        title="pw effect on success_mean",
        ylim=(0.0, 1.0),
    )

    # Diversity (only defined when >=2 valid trajectories exist).
    series_div = {
        "EnvA (steps44 baseline)": _series("diversity_valid_mean", "EnvA", sweep_label="A_pw_to2_steps44"),
        "EnvReplace (steps44 baseline)": _series("diversity_valid_mean", "EnvReplace"),
    }
    _plot_line(
        os.path.join(out_dir, "pw_vs_diversity_valid_mean.png"),
        series=series_div,
        xlabel="prior_weight_with_guide (pw)",
        ylabel="diversity_valid_mean (vendi score)",
        title="pw effect on trajectory diversity (valid trajectories only)",
        ylim=None,
    )

    # Low-pw zoomed comparison: use the steps44 baseline slice extracted from sweep_a_pw_v1.
    series_low = {
        "EnvA (from pw_v1, fixed other params)": _series("fraction_valid_min", "EnvA", sweep_label="A_pw_v1_steps44_baseline"),
        "EnvReplace (fixed other params)": _series("fraction_valid_min", "EnvReplace"),
    }
    _plot_line(
        os.path.join(out_dir, "pw_lowrange_vs_fraction_valid_min.png"),
        series=series_low,
        xlabel="prior_weight_with_guide (pw)",
        ylabel="fraction_valid_min (across seeds)",
        title="Low-pw region (EnvA baseline slice from pw_v1 vs EnvReplace): fraction_valid_min",
        ylim=(0.0, 1.0),
    )

    # Combined EnvA baseline curve: stitch pw_v1 (low pw) + pw_to2 (high pw) for a single EnvA-vs-EnvReplace plot.
    # This matches the user's request for comparing 0.95~1.5 (and beyond) while keeping other params fixed.
    def _map_by_pw(env_label: str, sweep_label: str) -> Dict[float, Dict[str, Any]]:
        out: Dict[float, Dict[str, Any]] = {}
        for r in agg:
            if r["env_label"] != env_label or r["sweep_label"] != sweep_label:
                continue
            out[float(r["pw"])] = r
        return out

    a_low = _map_by_pw("EnvA", "A_pw_v1_steps44_baseline")
    a_high = _map_by_pw("EnvA", "A_pw_to2_steps44")
    replace = _map_by_pw("EnvReplace", "Replace_pw_to2_steps44")

    def _stitch_envA(metric_key: str) -> Tuple[List[float], List[float]]:
        pws = sorted(set(a_low.keys()) | set(a_high.keys()))
        xs: List[float] = []
        ys: List[float] = []
        for pw in pws:
            src = a_high.get(pw) if pw in a_high else a_low.get(pw)
            if src is None:
                continue
            xs.append(pw)
            ys.append(float(src.get(metric_key, float("nan"))))
        return xs, ys

    def _series_replace(metric_key: str) -> Tuple[List[float], List[float]]:
        xs = sorted(replace.keys())
        ys = [float(replace[pw].get(metric_key, float("nan"))) for pw in xs]
        return xs, ys

    stitched_fv_min = {
        "EnvA (stitched, fixed other params)": _stitch_envA("fraction_valid_min"),
        "EnvReplace (fixed other params)": _series_replace("fraction_valid_min"),
    }
    _plot_line(
        os.path.join(out_dir, "pw_vs_fraction_valid_min_envA_stitched_vs_replace.png"),
        series=stitched_fv_min,
        xlabel="prior_weight_with_guide (pw)",
        ylabel="fraction_valid_min (across seeds)",
        title="EnvA vs EnvReplace (same steps44 baseline): fraction_valid_min",
        ylim=(0.0, 1.0),
    )

    stitched_div = {
        "EnvA (stitched, fixed other params)": _stitch_envA("diversity_valid_mean"),
        "EnvReplace (fixed other params)": _series_replace("diversity_valid_mean"),
    }
    _plot_line(
        os.path.join(out_dir, "pw_vs_diversity_envA_stitched_vs_replace.png"),
        series=stitched_div,
        xlabel="prior_weight_with_guide (pw)",
        ylabel="diversity_valid_mean (vendi score)",
        title="EnvA vs EnvReplace (same steps44 baseline): diversity_valid_mean",
        ylim=None,
    )

    # Generalization gap table (only pw values that exist on both sides).
    gap_rows: List[Dict[str, Any]] = []
    for pw in sorted(set(replace.keys()) & (set(a_low.keys()) | set(a_high.keys()))):
        a_src = a_high.get(pw) if pw in a_high else a_low.get(pw)
        r_src = replace.get(pw)
        if a_src is None or r_src is None:
            continue
        gap_rows.append(
            {
                "pw": pw,
                "envA_fraction_valid_min": float(a_src.get("fraction_valid_min", float("nan"))),
                "replace_fraction_valid_min": float(r_src.get("fraction_valid_min", float("nan"))),
                "gap_fraction_valid_min": float(a_src.get("fraction_valid_min", float("nan")))
                - float(r_src.get("fraction_valid_min", float("nan"))),
                "envA_fraction_valid_mean": float(a_src.get("fraction_valid_mean", float("nan"))),
                "replace_fraction_valid_mean": float(r_src.get("fraction_valid_mean", float("nan"))),
                "gap_fraction_valid_mean": float(a_src.get("fraction_valid_mean", float("nan")))
                - float(r_src.get("fraction_valid_mean", float("nan"))),
                "envA_diversity_mean": float(a_src.get("diversity_valid_mean", float("nan"))),
                "replace_diversity_mean": float(r_src.get("diversity_valid_mean", float("nan"))),
            }
        )
    if gap_rows:
        _write_csv(os.path.join(out_dir, "generalization_gap_by_pw.csv"), gap_rows)

    print(f"Wrote: {per_seed_csv}")
    print(f"Wrote: {agg_csv}")
    print(f"Wrote plots under: {out_dir}")


if __name__ == "__main__":
    main()
