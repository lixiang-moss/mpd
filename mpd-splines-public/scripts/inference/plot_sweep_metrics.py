import argparse
import csv
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib


matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and float(x) == float(x) and math.isfinite(float(x))


def _nanmean(values: Sequence[float]) -> float:
    xs = [v for v in values if v == v]
    if not xs:
        return float("nan")
    return sum(xs) / float(len(xs))


def _nanmin(values: Sequence[float]) -> float:
    xs = [v for v in values if v == v]
    if not xs:
        return float("nan")
    return min(xs)


def _as_tag_float(token: Optional[str]) -> Optional[float]:
    if token is None:
        return None
    s = token.strip()
    if not s:
        return None
    return float(s.replace("p", "."))


def _as_tag_int(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    s = token.strip()
    if not s:
        return None
    return int(s)


_DPM_TAG_RE = re.compile(
    r"^(?P<sampler>dpm_solver|dpm_solver_pp)_steps(?P<steps>\d+)_order(?P<order>\d+)_ct(?P<ct>[01])"
    r"(?:_m(?P<method>[^_]+)_skip(?P<skip>[^_]+)_sol(?P<sol>[^_]+)_lof(?P<lof>[01])_dz(?P<dz>[01]))?"
    r"(?:_pw(?P<pw>[0-9p]+))?"
    r"(?:_ng(?P<ng>\d+))?"
    r"(?:_glr(?P<glr>[0-9p]+))?"
    r"(?:_mpx(?P<mpx>[0-9p]+))?"
    r"(?:_tsg(?P<tsg>[0-9p]+))?"
    r"$"
)

_UNIPC_TAG_RE = re.compile(
    r"^(?P<sampler>unipc)_steps(?P<steps>\d+)_order(?P<order>\d+)_ct(?P<ct>[01])_var(?P<variant>.+)$"
)


def _parse_run_tag(run_tag: str) -> Dict[str, Any]:
    run_tag = run_tag.strip()
    m = _DPM_TAG_RE.match(run_tag)
    if m:
        return {
            "sampler": m.group("sampler"),
            "steps": int(m.group("steps")),
            "order": int(m.group("order")),
            "ct": int(m.group("ct")),
            "method": m.group("method"),
            "skip": m.group("skip"),
            "solver_type": m.group("sol"),
            "lower_order_final": _as_tag_int(m.group("lof")),
            "denoise_to_zero": _as_tag_int(m.group("dz")),
            "prior_weight_with_guide": _as_tag_float(m.group("pw")),
            "n_guide_steps": _as_tag_int(m.group("ng")),
            "guide_lr": _as_tag_float(m.group("glr")),
            "max_perturb_x": _as_tag_float(m.group("mpx")),
            "t_start_guide_steps_fraction": _as_tag_float(m.group("tsg")),
        }
    m = _UNIPC_TAG_RE.match(run_tag)
    if m:
        return {
            "sampler": m.group("sampler"),
            "steps": int(m.group("steps")),
            "order": int(m.group("order")),
            "ct": int(m.group("ct")),
            "variant": m.group("variant"),
        }
    return {"sampler": "unknown"}


@dataclass
class _Agg:
    n_rows: int = 0
    metric_sums: Dict[str, float] = field(default_factory=dict)
    metric_counts: Dict[str, int] = field(default_factory=dict)
    fraction_valid_min: float = float("nan")
    fraction_valid_max: float = float("nan")

    def add(self, row: Dict[str, str], metric_fields: Iterable[str]) -> None:
        self.n_rows += 1
        frac = _safe_float(row.get("fraction_valid", "nan"))
        if frac == frac:
            self.fraction_valid_min = frac if self.fraction_valid_min != self.fraction_valid_min else min(self.fraction_valid_min, frac)
            self.fraction_valid_max = frac if self.fraction_valid_max != self.fraction_valid_max else max(self.fraction_valid_max, frac)

        for f in metric_fields:
            v = _safe_float(row.get(f, "nan"))
            if v != v:
                continue
            self.metric_sums[f] = self.metric_sums.get(f, 0.0) + v
            self.metric_counts[f] = self.metric_counts.get(f, 0) + 1

    def mean(self, field: str) -> float:
        c = self.metric_counts.get(field, 0)
        if c <= 0:
            return float("nan")
        return self.metric_sums.get(field, 0.0) / float(c)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_scatter(
    out_path: str,
    xs: Sequence[float],
    ys: Sequence[float],
    cs: Optional[Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    c_label: str = "steps",
    xscale: str = "linear",
    yscale: str = "linear",
) -> None:
    plt.figure(figsize=(10, 6))
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs_arr) & np.isfinite(ys_arr)
    xs_arr = xs_arr[mask]
    ys_arr = ys_arr[mask]

    if cs is None:
        plt.scatter(xs_arr, ys_arr, s=10, alpha=0.45)
    else:
        cs_arr = np.asarray(cs, dtype=np.float64)[mask]
        sc = plt.scatter(xs_arr, ys_arr, c=cs_arr, cmap="viridis", s=12, alpha=0.55)
        cb = plt.colorbar(sc)
        cb.set_label(c_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xscale and xscale != "linear":
        plt.xscale(xscale)
    if yscale and yscale != "linear":
        plt.yscale(yscale)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_box(
    out_path: str,
    values_by_group: Dict[str, List[float]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    def _sort_key(label: str) -> Tuple[int, float, str]:
        if label in ("None", ""):
            return (2, float("inf"), label)
        try:
            return (0, float(label), label)
        except Exception:
            return (1, float("inf"), label)

    groups = sorted(values_by_group.keys(), key=_sort_key)
    data = []
    labels = []
    for g in groups:
        xs = [v for v in values_by_group[g] if v == v]
        if not xs:
            continue
        data.append(xs)
        labels.append(g)

    if not data:
        return

    plt.figure(figsize=(max(10, 1.2 * len(labels)), 6))
    try:
        plt.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_lines(
    out_path: str,
    x_values: Sequence[float],
    series: Dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: str = "linear",
) -> None:
    plt.figure(figsize=(10, 6))
    for label, ys in sorted(series.items(), key=lambda kv: kv[0]):
        ys_arr = np.asarray(list(ys), dtype=np.float64)
        plt.plot(x_values, ys_arr, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xscale and xscale != "linear":
        plt.xscale(xscale)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_heatmap(
    out_path: str,
    matrix: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    if matrix.size == 0 or not np.isfinite(matrix).any():
        return

    n_y, n_x = matrix.shape
    fig_w = max(10.0, 0.5 * float(n_x))
    fig_h = max(6.0, 0.45 * float(n_y))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(matrix, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    cb = plt.colorbar(im)
    cb.set_label("value")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ticks=list(range(n_x)), labels=list(x_labels), rotation=45, ha="right")
    plt.yticks(ticks=list(range(n_y)), labels=list(y_labels))
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_corr_heatmap(
    out_path: str,
    columns: Dict[str, Sequence[float]],
    title: str,
) -> None:
    keys = list(columns.keys())
    if len(keys) < 2:
        return

    data = []
    valid_keys = []
    for k in keys:
        xs = np.asarray(list(columns[k]), dtype=np.float64)
        if np.isfinite(xs).sum() < 5:
            continue
        if np.nanstd(xs) <= 1e-12:
            continue
        data.append(xs)
        valid_keys.append(k)

    if len(valid_keys) < 2:
        return

    X = np.stack(data, axis=0)  # (K, N)
    K, N = X.shape
    corr = np.full((K, K), np.nan, dtype=np.float64)
    for i in range(K):
        for j in range(K):
            xi = X[i]
            xj = X[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 5:
                continue
            a = xi[mask]
            b = xj[mask]
            a = a - a.mean()
            b = b - b.mean()
            denom = float(np.sqrt((a * a).mean()) * np.sqrt((b * b).mean()))
            if denom <= 1e-12:
                continue
            corr[i, j] = float((a * b).mean() / denom)

    if not np.isfinite(corr).any():
        return

    plt.figure(figsize=(max(10, 0.55 * float(K)), max(8, 0.55 * float(K))))
    im = plt.imshow(corr, origin="lower", vmin=-1.0, vmax=1.0, cmap="coolwarm")
    cb = plt.colorbar(im)
    cb.set_label("Pearson r")
    plt.xticks(ticks=list(range(K)), labels=valid_keys, rotation=45, ha="right")
    plt.yticks(ticks=list(range(K)), labels=valid_keys)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default="",
        help="Sweep results root (expects sweep_metrics.csv under this dir).",
    )
    parser.add_argument(
        "--csv_path",
        default="",
        help="Path to sweep_metrics.csv (overrides --results_root).",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory for plots and aggregated CSV (default: <results_root>/plots).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Print top-K configs by (fraction_valid_min, fraction_valid_mean, time).",
    )
    args = parser.parse_args()

    if not args.csv_path and not args.results_root:
        raise ValueError("Provide either --results_root or --csv_path.")

    csv_path = args.csv_path
    if not csv_path:
        csv_path = os.path.join(args.results_root, "sweep_metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    results_root = args.results_root or os.path.dirname(csv_path)
    out_dir = args.out_dir or os.path.join(results_root, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Read CSV and aggregate by run_tag across seeds.
    by_tag: Dict[str, _Agg] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # pick numeric metrics that exist
        candidate_metrics = [
            "success",
            "fraction_valid",
            "fraction_valid_no_joint_limits_vel_acc",
            "path_length_best",
            "path_length_valid_mean",
            "smoothness_best",
            "t_inference_total",
            "t_generator",
            "t_guide",
            "collision_intensity",
            "ee_pose_goal_error_position_norm_best",
            "ee_pose_goal_error_orientation_norm_best",
        ]
        metric_fields = [m for m in candidate_metrics if m in fieldnames]
        if "fraction_valid" not in metric_fields and "fraction_valid" in fieldnames:
            metric_fields.append("fraction_valid")

        for row in reader:
            tag = (row.get("run_tag") or "").strip()
            if not tag:
                continue
            agg = by_tag.get(tag)
            if agg is None:
                agg = _Agg()
                by_tag[tag] = agg
            agg.add(row, metric_fields=metric_fields)

    aggregated_rows: List[Dict[str, Any]] = []
    for tag, agg in by_tag.items():
        info = _parse_run_tag(tag)
        out: Dict[str, Any] = {
            "run_tag": tag,
            "n_rows": agg.n_rows,
            "fraction_valid_mean": agg.mean("fraction_valid"),
            "fraction_valid_min": agg.fraction_valid_min,
            "fraction_valid_max": agg.fraction_valid_max,
        }
        for f in metric_fields:
            out[f"{f}_mean"] = agg.mean(f)
        # parsed params
        for k, v in info.items():
            out[k] = v
        aggregated_rows.append(out)

    aggregated_csv = os.path.join(out_dir, "aggregated_by_run_tag.csv")
    _write_csv(aggregated_csv, aggregated_rows)

    # Print top-K by stability first.
    def _key_stable(r: Dict[str, Any]) -> Tuple:
        fv_min = r.get("fraction_valid_min", float("nan"))
        fv_mean = r.get("fraction_valid_mean", float("nan"))
        t = r.get("t_inference_total_mean", float("nan"))
        fv_min = fv_min if isinstance(fv_min, (int, float)) and fv_min == fv_min else -1.0
        fv_mean = fv_mean if isinstance(fv_mean, (int, float)) and fv_mean == fv_mean else -1.0
        t = t if isinstance(t, (int, float)) and t == t else float("inf")
        return (-fv_min, -fv_mean, t)

    top = sorted(aggregated_rows, key=_key_stable)[: max(1, args.top_k)]
    top_txt = os.path.join(out_dir, "top_configs.txt")
    with open(top_txt, "w") as f:
        for r in top:
            f.write(
                f"{r['run_tag']}\n"
                f"  fraction_valid_min={r.get('fraction_valid_min'):.3f} "
                f"fraction_valid_mean={r.get('fraction_valid_mean'):.3f} "
                f"t_total_mean={r.get('t_inference_total_mean', float('nan')):.3f}\n"
            )

    # Build arrays for plotting.
    steps = [r.get("steps") if isinstance(r.get("steps"), int) else float("nan") for r in aggregated_rows]
    frac_mean = [r.get("fraction_valid_mean", float("nan")) for r in aggregated_rows]
    frac_min = [r.get("fraction_valid_min", float("nan")) for r in aggregated_rows]
    t_total = [r.get("t_inference_total_mean", float("nan")) for r in aggregated_rows]
    smooth = [r.get("smoothness_best_mean", float("nan")) for r in aggregated_rows]
    collision = [r.get("collision_intensity_mean", float("nan")) for r in aggregated_rows]
    pw = [r.get("prior_weight_with_guide") if _is_finite_number(r.get("prior_weight_with_guide")) else float("nan") for r in aggregated_rows]
    glr = [r.get("guide_lr") if _is_finite_number(r.get("guide_lr")) else float("nan") for r in aggregated_rows]
    tsg = [
        r.get("t_start_guide_steps_fraction") if _is_finite_number(r.get("t_start_guide_steps_fraction")) else float("nan")
        for r in aggregated_rows
    ]

    _plot_scatter(
        os.path.join(out_dir, "scatter_fraction_mean_vs_time.png"),
        xs=t_total,
        ys=frac_mean,
        cs=steps,
        xlabel="t_inference_total_mean (s)",
        ylabel="fraction_valid_mean",
        title="fraction_valid_mean vs time (colored by steps)",
        c_label="steps",
    )
    _plot_scatter(
        os.path.join(out_dir, "scatter_fraction_min_vs_time.png"),
        xs=t_total,
        ys=frac_min,
        cs=steps,
        xlabel="t_inference_total_mean (s)",
        ylabel="fraction_valid_min",
        title="fraction_valid_min vs time (colored by steps)",
        c_label="steps",
    )
    _plot_scatter(
        os.path.join(out_dir, "scatter_fraction_mean_vs_smoothness.png"),
        xs=smooth,
        ys=frac_mean,
        cs=steps,
        xlabel="smoothness_best_mean",
        ylabel="fraction_valid_mean",
        title="fraction_valid_mean vs smoothness (colored by steps)",
        c_label="steps",
    )
    if any(isinstance(c, (int, float)) and c == c for c in collision):
        _plot_scatter(
            os.path.join(out_dir, "scatter_fraction_mean_vs_collision.png"),
            xs=collision,
            ys=frac_mean,
            cs=steps,
            xlabel="collision_intensity_mean",
            ylabel="fraction_valid_mean",
            title="fraction_valid_mean vs collision_intensity (colored by steps)",
            c_label="steps",
        )

    # Box plots for key hyperparams (when present).
    def _group_box(param_key: str, metric_key: str) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for r in aggregated_rows:
            v = r.get(param_key)
            if v is None:
                continue
            key = str(v)
            out.setdefault(key, []).append(float(r.get(metric_key, float("nan"))))
        return out

    by_glr = _group_box("guide_lr", "fraction_valid_mean")
    _plot_box(
        os.path.join(out_dir, "box_fraction_mean_by_guide_lr.png"),
        values_by_group=by_glr,
        xlabel="guide_lr",
        ylabel="fraction_valid_mean",
        title="fraction_valid_mean grouped by guide_lr",
    )
    by_pw = _group_box("prior_weight_with_guide", "fraction_valid_mean")
    _plot_box(
        os.path.join(out_dir, "box_fraction_mean_by_prior_weight.png"),
        values_by_group=by_pw,
        xlabel="prior_weight_with_guide",
        ylabel="fraction_valid_mean",
        title="fraction_valid_mean grouped by prior_weight_with_guide",
    )
    by_tsg = _group_box("t_start_guide_steps_fraction", "fraction_valid_mean")
    _plot_box(
        os.path.join(out_dir, "box_fraction_mean_by_t_start_guide.png"),
        values_by_group=by_tsg,
        xlabel="t_start_guide_steps_fraction",
        ylabel="fraction_valid_mean",
        title="fraction_valid_mean grouped by t_start_guide_steps_fraction",
    )

    # Additional plots focused on stability (fraction_valid_min).
    by_glr_min = _group_box("guide_lr", "fraction_valid_min")
    _plot_box(
        os.path.join(out_dir, "box_fraction_min_by_guide_lr.png"),
        values_by_group=by_glr_min,
        xlabel="guide_lr",
        ylabel="fraction_valid_min",
        title="fraction_valid_min grouped by guide_lr",
    )
    by_pw_min = _group_box("prior_weight_with_guide", "fraction_valid_min")
    _plot_box(
        os.path.join(out_dir, "box_fraction_min_by_prior_weight.png"),
        values_by_group=by_pw_min,
        xlabel="prior_weight_with_guide",
        ylabel="fraction_valid_min",
        title="fraction_valid_min grouped by prior_weight_with_guide",
    )
    by_tsg_min = _group_box("t_start_guide_steps_fraction", "fraction_valid_min")
    _plot_box(
        os.path.join(out_dir, "box_fraction_min_by_t_start_guide.png"),
        values_by_group=by_tsg_min,
        xlabel="t_start_guide_steps_fraction",
        ylabel="fraction_valid_min",
        title="fraction_valid_min grouped by t_start_guide_steps_fraction",
    )
    by_steps_min = _group_box("steps", "fraction_valid_min")
    _plot_box(
        os.path.join(out_dir, "box_fraction_min_by_steps.png"),
        values_by_group=by_steps_min,
        xlabel="steps",
        ylabel="fraction_valid_min",
        title="fraction_valid_min grouped by steps",
    )
    by_steps_time = _group_box("steps", "t_inference_total_mean")
    _plot_box(
        os.path.join(out_dir, "box_time_by_steps.png"),
        values_by_group=by_steps_time,
        xlabel="steps",
        ylabel="t_inference_total_mean (s)",
        title="t_inference_total_mean grouped by steps",
    )

    # Generic box plots for other (possibly categorical) knobs (only if >= 2 groups).
    for key in [
        "sampler",
        "ct",
        "order",
        "method",
        "skip",
        "solver_type",
        "denoise_to_zero",
        "lower_order_final",
        "n_guide_steps",
        "max_perturb_x",
        "variant",
    ]:
        grouped = _group_box(key, "fraction_valid_min")
        if len(grouped) < 2:
            continue
        _plot_box(
            os.path.join(out_dir, f"box_fraction_min_by_{key}.png"),
            values_by_group=grouped,
            xlabel=key,
            ylabel="fraction_valid_min",
            title=f"fraction_valid_min grouped by {key}",
        )

    # Scatter plots: hyperparams vs stability.
    if any(_is_finite_number(v) for v in pw):
        _plot_scatter(
            os.path.join(out_dir, "scatter_fraction_min_vs_prior_weight.png"),
            xs=pw,
            ys=frac_min,
            cs=steps,
            xlabel="prior_weight_with_guide",
            ylabel="fraction_valid_min",
            title="fraction_valid_min vs prior_weight_with_guide (colored by steps)",
            c_label="steps",
        )
        _plot_scatter(
            os.path.join(out_dir, "scatter_fraction_min_vs_steps_color_prior_weight.png"),
            xs=steps,
            ys=frac_min,
            cs=pw,
            xlabel="steps",
            ylabel="fraction_valid_min",
            title="fraction_valid_min vs steps (colored by prior_weight_with_guide)",
            c_label="prior_weight_with_guide",
        )
        _plot_scatter(
            os.path.join(out_dir, "scatter_time_vs_prior_weight.png"),
            xs=pw,
            ys=t_total,
            cs=steps,
            xlabel="prior_weight_with_guide",
            ylabel="t_inference_total_mean (s)",
            title="time vs prior_weight_with_guide (colored by steps)",
            c_label="steps",
        )

    if any(_is_finite_number(v) for v in glr):
        _plot_scatter(
            os.path.join(out_dir, "scatter_fraction_min_vs_guide_lr_linear.png"),
            xs=glr,
            ys=frac_min,
            cs=steps,
            xlabel="guide_lr",
            ylabel="fraction_valid_min",
            title="fraction_valid_min vs guide_lr (linear x, colored by steps)",
            c_label="steps",
        )
        if all((v > 0.0) for v in glr if _is_finite_number(v)):
            _plot_scatter(
                os.path.join(out_dir, "scatter_fraction_min_vs_guide_lr_logx.png"),
                xs=glr,
                ys=frac_min,
                cs=steps,
                xlabel="guide_lr",
                ylabel="fraction_valid_min",
                title="fraction_valid_min vs guide_lr (log x, colored by steps)",
                c_label="steps",
                xscale="log",
            )
        _plot_scatter(
            os.path.join(out_dir, "scatter_time_vs_guide_lr_linear.png"),
            xs=glr,
            ys=t_total,
            cs=steps,
            xlabel="guide_lr",
            ylabel="t_inference_total_mean (s)",
            title="time vs guide_lr (colored by steps)",
            c_label="steps",
        )

    if any(_is_finite_number(v) for v in tsg):
        _plot_scatter(
            os.path.join(out_dir, "scatter_fraction_min_vs_t_start_guide.png"),
            xs=tsg,
            ys=frac_min,
            cs=steps,
            xlabel="t_start_guide_steps_fraction",
            ylabel="fraction_valid_min",
            title="fraction_valid_min vs t_start_guide_steps_fraction (colored by steps)",
            c_label="steps",
        )

    # Line plots: for each steps, best fraction_valid_min over the remaining params.
    def _unique_sorted_floats(values: Sequence[Any]) -> List[float]:
        xs: List[float] = []
        for v in values:
            if _is_finite_number(v):
                xs.append(float(v))
        xs = sorted(set(xs))
        return xs

    def _best_series_over(
        x_key: str,
        split_key: str,
        metric_key: str,
    ) -> Tuple[List[float], Dict[str, List[float]]]:
        x_vals = _unique_sorted_floats([r.get(x_key) for r in aggregated_rows])
        split_vals = _unique_sorted_floats([r.get(split_key) for r in aggregated_rows])
        if len(x_vals) < 2 or len(split_vals) < 1:
            return ([], {})

        best: Dict[Tuple[float, float], float] = {}
        for r in aggregated_rows:
            xv = r.get(x_key)
            sv = r.get(split_key)
            mv = r.get(metric_key)
            if not (_is_finite_number(xv) and _is_finite_number(sv) and _is_finite_number(mv)):
                continue
            key = (float(sv), float(xv))
            cur = best.get(key, float("nan"))
            if cur != cur or float(mv) > cur:
                best[key] = float(mv)

        series: Dict[str, List[float]] = {}
        for sv in split_vals:
            ys = []
            for xv in x_vals:
                ys.append(best.get((sv, xv), float("nan")))
            series[f"{split_key}={sv:g}"] = ys
        return (x_vals, series)

    if by_pw_min and len(set(v for v in pw if _is_finite_number(v))) >= 2:
        x_vals, series = _best_series_over("prior_weight_with_guide", "steps", "fraction_valid_min")
        if x_vals and series:
            _plot_lines(
                os.path.join(out_dir, "line_best_fraction_min_vs_prior_weight_by_steps.png"),
                x_values=x_vals,
                series=series,
                xlabel="prior_weight_with_guide",
                ylabel="best fraction_valid_min (over other params)",
                title="Best fraction_valid_min vs prior_weight_with_guide, split by steps",
            )

    if by_glr_min and len(set(v for v in glr if _is_finite_number(v))) >= 2:
        x_vals, series = _best_series_over("guide_lr", "steps", "fraction_valid_min")
        if x_vals and series and all(v > 0.0 for v in x_vals):
            _plot_lines(
                os.path.join(out_dir, "line_best_fraction_min_vs_guide_lr_by_steps_logx.png"),
                x_values=x_vals,
                series=series,
                xlabel="guide_lr",
                ylabel="best fraction_valid_min (over other params)",
                title="Best fraction_valid_min vs guide_lr (log x), split by steps",
                xscale="log",
            )

    if by_tsg_min and len(set(v for v in tsg if _is_finite_number(v))) >= 2:
        x_vals, series = _best_series_over("t_start_guide_steps_fraction", "steps", "fraction_valid_min")
        if x_vals and series:
            _plot_lines(
                os.path.join(out_dir, "line_best_fraction_min_vs_t_start_guide_by_steps.png"),
                x_values=x_vals,
                series=series,
                xlabel="t_start_guide_steps_fraction",
                ylabel="best fraction_valid_min (over other params)",
                title="Best fraction_valid_min vs t_start_guide_steps_fraction, split by steps",
            )

    # Heatmap: best fraction_valid_min over (pw, guide_lr) for each steps (if steps list not huge).
    steps_unique = sorted({int(s) for s in steps if isinstance(s, int)})
    if steps_unique and len(steps_unique) <= 12 and by_pw_min and by_glr_min:
        pw_vals = _unique_sorted_floats([r.get("prior_weight_with_guide") for r in aggregated_rows])
        glr_vals = _unique_sorted_floats([r.get("guide_lr") for r in aggregated_rows])
        pw_index = {v: i for i, v in enumerate(pw_vals)}
        glr_index = {v: i for i, v in enumerate(glr_vals)}
        if pw_vals and glr_vals:
            for s in steps_unique:
                mat = np.full((len(pw_vals), len(glr_vals)), np.nan, dtype=np.float64)
                for r in aggregated_rows:
                    if r.get("steps") != s:
                        continue
                    pwv = r.get("prior_weight_with_guide")
                    glrv = r.get("guide_lr")
                    mv = r.get("fraction_valid_min")
                    if not (_is_finite_number(pwv) and _is_finite_number(glrv) and _is_finite_number(mv)):
                        continue
                    iy = pw_index.get(float(pwv))
                    ix = glr_index.get(float(glrv))
                    if iy is None or ix is None:
                        continue
                    cur = mat[iy, ix]
                    if cur != cur or float(mv) > cur:
                        mat[iy, ix] = float(mv)

                _plot_heatmap(
                    os.path.join(out_dir, f"heatmap_best_fraction_min_pw_vs_glr_steps{s}.png"),
                    matrix=mat,
                    x_labels=[f"{v:g}" for v in glr_vals],
                    y_labels=[f"{v:g}" for v in pw_vals],
                    xlabel="guide_lr",
                    ylabel="prior_weight_with_guide",
                    title=f"Best fraction_valid_min over (pw, guide_lr) @ steps={s}",
                    vmin=0.0,
                    vmax=1.0,
                )

    # Correlation overview (numeric columns only).
    corr_cols: Dict[str, Sequence[float]] = {
        "fraction_valid_min": frac_min,
        "fraction_valid_mean": frac_mean,
        "t_total_mean": t_total,
        "steps": [float(s) if isinstance(s, int) else float("nan") for s in steps],
        "pw": pw,
        "guide_lr": glr,
        "t_start_guide": tsg,
        "smooth_best_mean": smooth,
        "collision_mean": collision,
    }
    _plot_corr_heatmap(
        os.path.join(out_dir, "corr_overview_numeric.png"),
        columns=corr_cols,
        title="Correlation overview (aggregated_by_run_tag rows)",
    )

    # Line plot: best fraction_valid_mean vs steps for each t_start_guide_steps_fraction,
    # within the empirically good region (pw=1.05, guide_lr=0.01) if those fields exist.
    if by_pw and by_glr and by_tsg:
        # Find if the common good region exists
        focus_pw = 1.05 if "1.05" in by_pw else None
        focus_glr = 0.01 if "0.01" in by_glr else None
        if focus_pw is not None and focus_glr is not None:
            tsg_values = sorted(float(k) for k in by_tsg.keys())
            steps_values = sorted({int(r["steps"]) for r in aggregated_rows if isinstance(r.get("steps"), int)})

            series: Dict[str, List[float]] = {}
            for tsg in tsg_values:
                ys = []
                for s in steps_values:
                    candidates = [
                        r
                        for r in aggregated_rows
                        if r.get("steps") == s
                        and r.get("prior_weight_with_guide") == focus_pw
                        and r.get("guide_lr") == focus_glr
                        and r.get("t_start_guide_steps_fraction") == tsg
                    ]
                    if not candidates:
                        ys.append(float("nan"))
                        continue
                    best = max(candidates, key=lambda rr: rr.get("fraction_valid_mean", float("nan")))
                    ys.append(best.get("fraction_valid_mean", float("nan")))
                series[f"tsg={tsg:g}"] = ys

            _plot_lines(
                os.path.join(out_dir, "line_best_fraction_mean_vs_steps_by_tsg_pw1p05_glr0p01.png"),
                x_values=steps_values,
                series=series,
                xlabel="dpm_solver_steps",
                ylabel="best fraction_valid_mean",
                title="Best fraction_valid_mean vs steps (pw=1.05, guide_lr=0.01), split by t_start_guide_steps_fraction",
            )

    print(f"Wrote aggregated CSV: {aggregated_csv}")
    print(f"Wrote plots under: {out_dir}")
    print(f"Wrote top configs: {top_txt}")


if __name__ == "__main__":
    main()
