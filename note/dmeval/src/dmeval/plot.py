from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dmeval.utils import is_nan, to_float


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _label(row: Dict[str, Any]) -> str:
    run_tag = str(row.get("run_tag", ""))
    seed = str(row.get("seed", ""))
    return f"{run_tag}/{seed}" if seed else run_tag


def _bar_plot(rows: List[Dict[str, Any]], *, y_key: str, yerr_key: str, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("Plotting requires matplotlib. Install dmeval with the 'plot' extra: dmeval[plot].") from exc

    labels = [_label(r) for r in rows]
    ys = [to_float(r.get(y_key, float("nan"))) for r in rows]
    yerrs = [to_float(r.get(yerr_key, float("nan"))) for r in rows]
    yerrs = [0.0 if is_nan(e) else e for e in yerrs]

    fig_w = max(8.0, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    xs = list(range(len(labels)))
    ax.bar(xs, ys, yerr=yerrs, capsize=3)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_plot(
    rows: List[Dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("Plotting requires matplotlib. Install dmeval with the 'plot' extra: dmeval[plot].") from exc

    xs = [to_float(r.get(x_key, float("nan"))) for r in rows]
    ys = [to_float(r.get(y_key, float("nan"))) for r in rows]
    labels = [_label(r) for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(xs, ys)
    for x, y, lbl in zip(xs, ys, labels):
        if is_nan(x) or is_nan(y):
            continue
        ax.annotate(lbl, (x, y), fontsize=8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_from_run_metrics(*, run_metrics: Path, out_dir: Path, scenario: str = "") -> None:
    rows = _load_rows(run_metrics)
    if scenario:
        rows = [r for r in rows if str(r.get("scenario", "")) == scenario]
    if not rows:
        raise RuntimeError("No rows to plot.")

    by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_scene[str(r.get("scenario", ""))].append(r)

    for scene, scene_rows in by_scene.items():
        scene_rows = sorted(scene_rows, key=lambda r: (_label(r)))
        _bar_plot(
            scene_rows,
            y_key="success_mean",
            yerr_key="success_std",
            out_path=out_dir / f"{scene}_bar_success.png",
            title=f"{scene} - success_mean",
        )
        _bar_plot(
            scene_rows,
            y_key="fraction_valid_mean",
            yerr_key="fraction_valid_std",
            out_path=out_dir / f"{scene}_bar_fraction_valid.png",
            title=f"{scene} - fraction_valid_mean",
        )
        _bar_plot(
            scene_rows,
            y_key="t_inference_total_mean",
            yerr_key="t_inference_total_std",
            out_path=out_dir / f"{scene}_bar_time.png",
            title=f"{scene} - t_inference_total_mean (sec)",
        )
        _scatter_plot(
            scene_rows,
            x_key="t_inference_total_mean",
            y_key="success_mean",
            out_path=out_dir / f"{scene}_scatter_time_vs_success.png",
            title=f"{scene} - time vs success",
            xlabel="t_inference_total_mean (sec)",
            ylabel="success_mean",
        )


def plot_command(*, run_metrics: str, out_dir: str, scenario: str) -> int:
    try:
        plot_from_run_metrics(run_metrics=Path(run_metrics), out_dir=Path(out_dir), scenario=(scenario or ""))
    except Exception as exc:
        print(f"[dmeval] ERROR: {exc}")
        return 1
    print(f"[dmeval] Wrote plots to {out_dir}")
    return 0

