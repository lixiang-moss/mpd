import argparse
import copy
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


def _parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_bool_list(raw: str) -> List[bool]:
    if not raw:
        return []
    out: List[bool] = []
    for item in raw.split(","):
        s = item.strip().lower()
        if not s:
            continue
        if s in ("1", "true", "t", "yes", "y", "on"):
            out.append(True)
        elif s in ("0", "false", "f", "no", "n", "off"):
            out.append(False)
        else:
            raise ValueError(f"Unsupported bool token: {item!r} (expected true/false)")
    return out


def _parse_float_list(raw: str) -> List[float]:
    if not raw:
        return []
    out: List[float] = []
    for item in raw.split(","):
        s = item.strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _time_modes_from_arg(mode: str) -> List[bool]:
    if mode == "both":
        return [False, True]
    if mode == "continuous":
        return [True]
    if mode == "discrete":
        return [False]
    raise ValueError(f"Unsupported time mode: {mode}")


def _run_command(cmd: Sequence[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(list(cmd), check=True)


def _has_results(results_root: str, run_tag: str, seed: int) -> bool:
    seed_dir = os.path.join(results_root, run_tag, str(seed))
    if not os.path.isdir(seed_dir):
        return False
    for fname in os.listdir(seed_dir):
        if fname.startswith("results_single_plan-") and fname.endswith(".pt"):
            return True
    return False


def _write_yaml(cfg: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


COARSE_TAG_RE = re.compile(r"^(dpm_solver|dpm_solver_pp)_steps(?P<steps>\d+)_order(?P<order>\d+)_ct(?P<ct>[01])$")


@dataclass(frozen=True)
class AggregatedMetrics:
    run_tag: str
    n_rows: int
    success_mean: float
    fraction_valid_mean: float
    path_length_best_mean: float
    smoothness_best_mean: float
    t_inference_total_mean: float


def _safe_float(raw: str) -> float:
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _nanmean(values: Sequence[float]) -> float:
    xs = [v for v in values if v == v]  # v==v filters NaN
    if not xs:
        return float("nan")
    return sum(xs) / float(len(xs))


def _aggregate_csv(csv_path: str) -> List[AggregatedMetrics]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing metrics CSV: {csv_path}. Run collect_sweep_metrics.py first (or pass --collect_metrics)."
        )

    by_tag: Dict[str, List[Dict[str, float]]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_tag = (row.get("run_tag") or "").strip()
            if not run_tag:
                continue
            by_tag.setdefault(run_tag, []).append(
                {
                    "success": _safe_float(row.get("success", "")),
                    "fraction_valid": _safe_float(row.get("fraction_valid", "")),
                    "path_length_best": _safe_float(row.get("path_length_best", "")),
                    "smoothness_best": _safe_float(row.get("smoothness_best", "")),
                    "t_inference_total": _safe_float(row.get("t_inference_total", "")),
                }
            )

    out: List[AggregatedMetrics] = []
    for run_tag, rows in by_tag.items():
        out.append(
            AggregatedMetrics(
                run_tag=run_tag,
                n_rows=len(rows),
                success_mean=_nanmean([r["success"] for r in rows]),
                fraction_valid_mean=_nanmean([r["fraction_valid"] for r in rows]),
                path_length_best_mean=_nanmean([r["path_length_best"] for r in rows]),
                smoothness_best_mean=_nanmean([r["smoothness_best"] for r in rows]),
                t_inference_total_mean=_nanmean([r["t_inference_total"] for r in rows]),
            )
        )
    return out


def _select_topk_coarse_runs(
    aggregated: Sequence[AggregatedMetrics],
    top_k: int,
) -> List[AggregatedMetrics]:
    coarse_only = [m for m in aggregated if COARSE_TAG_RE.match(m.run_tag)]

    def key(m: AggregatedMetrics) -> Tuple:
        # Lexicographic "better" ordering:
        # 1) success_mean desc
        # 2) fraction_valid_mean desc
        # 3) path_length_best_mean asc
        # 4) smoothness_best_mean asc
        # 5) t_inference_total_mean asc
        # NaNs are treated as worst.
        success = m.success_mean if m.success_mean == m.success_mean else -1.0
        fraction = m.fraction_valid_mean if m.fraction_valid_mean == m.fraction_valid_mean else -1.0
        path = m.path_length_best_mean if m.path_length_best_mean == m.path_length_best_mean else float("inf")
        smooth = m.smoothness_best_mean if m.smoothness_best_mean == m.smoothness_best_mean else float("inf")
        t = m.t_inference_total_mean if m.t_inference_total_mean == m.t_inference_total_mean else float("inf")
        return (-success, -fraction, path, smooth, t)

    ranked = sorted(coarse_only, key=key)
    return ranked[: max(0, top_k)]


def _parse_coarse_tag(run_tag: str) -> Tuple[str, int, int, bool]:
    m = COARSE_TAG_RE.match(run_tag)
    if not m:
        raise ValueError(f"Not a coarse run_tag: {run_tag}")
    prefix = m.group(1)
    steps = int(m.group("steps"))
    order = int(m.group("order"))
    use_cont_time = bool(int(m.group("ct")))
    return prefix, steps, order, use_cont_time


def _iter_product(*seqs: Iterable) -> Iterable[Tuple]:
    out: List[Tuple] = [()]
    for seq in seqs:
        out = [prev + (x,) for prev in out for x in seq]
    return out


def _format_float_tag(value: float, precision: int = 4) -> str:
    text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", required=True, help="Path to base inference YAML config.")
    parser.add_argument("--planner_alg", default="", help="Override planner_alg in generated configs (e.g., mpd).")
    parser.add_argument("--output_dir", default="mpd-splines-public/scripts/inference/cfgs/generated")
    parser.add_argument("--results_root", default="mpd-splines-public/scripts/inference/logs/sweep_dpm_solver")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="2", help="Comma-separated list of seeds, e.g. 0,1,2")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["coarse", "refine", "all"],
        help="coarse: steps/order/time/solver-vs-solver++; refine: extra knobs on top-K coarse; all: coarse then refine",
    )
    parser.add_argument("--generate_only", action="store_true", help="Only generate configs, do not run inference.")
    parser.add_argument(
        "--skip_existing",
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip runs that already have results_single_plan-*.pt under results_root (default: enabled).",
    )
    parser.add_argument(
        "--no_skip_existing",
        "--no-skip_existing",
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Disable skipping existing runs.",
    )
    parser.add_argument(
        "--collect_metrics",
        action="store_true",
        help="Run collect_sweep_metrics.py (required for refine selection unless you already ran it).",
    )

    # ------------------------------ coarse grid ------------------------------ #
    parser.add_argument("--coarse_steps", default="15,20,26,32,40", help="Comma-separated steps list.")
    parser.add_argument("--coarse_orders", default="2,3", help="Comma-separated order list.")
    parser.add_argument(
        "--coarse_time_modes",
        default="both",
        choices=["continuous", "discrete", "both"],
        help="Evaluate continuous time, discrete time, or both.",
    )
    parser.add_argument(
        "--coarse_compare_solver_pp",
        "--coarse-compare-solver-pp",
        dest="coarse_compare_solver_pp",
        action="store_true",
        default=True,
        help="Compare DPM-Solver (dpmsolver) vs DPM-Solver++ (dpmsolver++) (default: enabled).",
    )
    parser.add_argument(
        "--no_coarse_compare_solver_pp",
        "--no-coarse_compare_solver_pp",
        "--no-coarse-compare-solver-pp",
        dest="coarse_compare_solver_pp",
        action="store_false",
        help="Disable comparing solver vs solver++ (only run solver++).",
    )

    # ------------------------------ refine grid ------------------------------ #
    parser.add_argument("--top_k", type=int, default=3, help="Top-K coarse configs to refine.")
    parser.add_argument(
        "--refine_methods",
        default="multistep,singlestep",
        help="Comma-separated DPM-Solver methods to sweep in refine stage.",
    )
    parser.add_argument(
        "--refine_skip_types",
        default="logSNR",
        help="Comma-separated skip_type list in refine stage (e.g., logSNR,time_uniform).",
    )
    parser.add_argument(
        "--refine_solver_types",
        default="dpmsolver",
        help="Comma-separated solver_type list in refine stage (e.g., dpmsolver,taylor).",
    )
    parser.add_argument("--refine_lower_order_final", default="true,false", help="Comma-separated bool list.")
    parser.add_argument("--refine_denoise_to_zero", default="false,true", help="Comma-separated bool list.")
    parser.add_argument(
        "--refine_prior_weights",
        default="",
        help="Comma-separated prior_weight_with_guide list (empty: use base config value).",
    )
    parser.add_argument(
        "--refine_n_guide_steps",
        default="",
        help="Comma-separated n_guide_steps list (empty: use base config value).",
    )
    parser.add_argument(
        "--refine_guide_lr",
        default="",
        help="Comma-separated guide_lr list (empty: use base config value).",
    )
    parser.add_argument(
        "--refine_max_perturb_x",
        default="",
        help="Comma-separated max_perturb_x list (empty: use base config value).",
    )
    parser.add_argument(
        "--refine_t_start_guide_steps_fraction",
        default="",
        help="Comma-separated t_start_guide_steps_fraction list (empty: use base config value).",
    )

    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed.")

    with open(args.base_cfg, "r") as f:
        base_cfg = yaml.safe_load(f)
    if "dpm_solver" not in base_cfg:
        raise ValueError("Base config missing 'dpm_solver' section.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)

    def run_inference(run_tag: str, cfg_path: str) -> None:
        results_dir = os.path.join(args.results_root, run_tag)
        for seed in seeds:
            if args.skip_existing and _has_results(args.results_root, run_tag, seed):
                print(f"Skip existing: run_tag={run_tag} seed={seed}")
                continue
            cmd = [
                "python3",
                "mpd-splines-public/scripts/inference/inference.py",
                "--cfg_inference_path",
                cfg_path,
                "--results_dir",
                results_dir,
                "--seed",
                str(seed),
                "--save_results_single_plan_low_mem",
                "true",
                "--device",
                args.device,
            ]
            _run_command(cmd)

    def collect_metrics() -> None:
        cmd = [
            "python3",
            "mpd-splines-public/scripts/inference/collect_sweep_metrics.py",
            "--results_root",
            args.results_root,
        ]
        _run_command(cmd)

    # --------------------------------------------------------------------- #
    # Phase: coarse (steps/order/time + solver vs solver++)
    # --------------------------------------------------------------------- #
    if args.phase in ("coarse", "all"):
        coarse_steps = _parse_int_list(args.coarse_steps)
        coarse_orders = _parse_int_list(args.coarse_orders)
        coarse_time_modes = _time_modes_from_arg(args.coarse_time_modes)
        if not coarse_steps or not coarse_orders:
            raise ValueError("Both --coarse_steps and --coarse_orders must contain at least one value.")

        algo_types = ["dpmsolver++", "dpmsolver"] if args.coarse_compare_solver_pp else ["dpmsolver++"]

        combos = _iter_product(coarse_steps, coarse_orders, coarse_time_modes, algo_types)
        for steps, order, use_cont_time, algorithm_type in combos:
            cfg = copy.deepcopy(base_cfg)
            if args.planner_alg:
                cfg["planner_alg"] = args.planner_alg

            diffusion_method = "dpm_solver_pp" if algorithm_type == "dpmsolver++" else "dpm_solver"
            cfg["diffusion_sampling_method"] = diffusion_method

            cfg["dpm_solver"]["dpm_solver_steps"] = steps
            cfg["dpm_solver"]["dpm_solver_order"] = order
            cfg["dpm_solver"]["use_continuous_time"] = bool(use_cont_time)
            cfg["dpm_solver"]["dpm_solver_algorithm_type"] = algorithm_type

            run_tag = f"{diffusion_method}_steps{steps}_order{order}_ct{int(use_cont_time)}"

            cfg_name = os.path.splitext(os.path.basename(args.base_cfg))[0]
            cfg_path = os.path.join(args.output_dir, f"{cfg_name}__{run_tag}.yaml")
            _write_yaml(cfg, cfg_path)

            if args.generate_only:
                print(f"Generated config: {cfg_path}")
                continue

            run_inference(run_tag, cfg_path)

        if (args.collect_metrics or args.phase == "all") and not args.generate_only:
            collect_metrics()

    # --------------------------------------------------------------------- #
    # Phase: refine (extra knobs on top-K coarse configs)
    # --------------------------------------------------------------------- #
    if args.phase in ("refine", "all"):
        if args.generate_only:
            raise ValueError("refine phase requires running coarse + collecting metrics; --generate_only is unsupported.")

        if args.collect_metrics and args.phase == "refine":
            # Ensure metrics are up-to-date before selecting top-K.
            collect_metrics()

        csv_path = os.path.join(args.results_root, "sweep_metrics.csv")
        aggregated = _aggregate_csv(csv_path)
        topk = _select_topk_coarse_runs(aggregated, top_k=args.top_k)
        if not topk:
            raise RuntimeError(
                "No coarse DPM-Solver runs found to refine. "
                "Expected run_tag like dpm_solver_stepsXX_orderY_ctZ under results_root."
            )

        refine_methods = _parse_str_list(args.refine_methods)
        refine_skip_types = _parse_str_list(args.refine_skip_types)
        refine_solver_types = _parse_str_list(args.refine_solver_types)
        refine_lof = _parse_bool_list(args.refine_lower_order_final)
        refine_dz = _parse_bool_list(args.refine_denoise_to_zero)
        refine_prior_weights = _parse_float_list(args.refine_prior_weights)
        refine_n_guide_steps = _parse_int_list(args.refine_n_guide_steps)
        refine_guide_lr = _parse_float_list(args.refine_guide_lr)
        refine_max_perturb_x = _parse_float_list(args.refine_max_perturb_x)
        refine_t_start_guide_steps_fraction = _parse_float_list(args.refine_t_start_guide_steps_fraction)

        base_dpm = base_cfg.get("dpm_solver", {})
        if not refine_prior_weights:
            refine_prior_weights = [float(base_dpm.get("prior_weight_with_guide", 1.0))]
        if not refine_n_guide_steps:
            refine_n_guide_steps = [int(base_dpm.get("n_guide_steps", 1))]
        if not refine_guide_lr:
            refine_guide_lr = [float(base_dpm.get("guide_lr", 0.05))]
        if not refine_max_perturb_x:
            refine_max_perturb_x = [float(base_dpm.get("max_perturb_x", 0.1))]
        if not refine_t_start_guide_steps_fraction:
            refine_t_start_guide_steps_fraction = [float(base_dpm.get("t_start_guide_steps_fraction", 0.0))]

        if (
            not refine_methods
            or not refine_skip_types
            or not refine_solver_types
            or not refine_lof
            or not refine_dz
            or not refine_prior_weights
            or not refine_n_guide_steps
            or not refine_guide_lr
            or not refine_max_perturb_x
            or not refine_t_start_guide_steps_fraction
        ):
            raise ValueError("Refine grids must be non-empty.")

        print("Top-K coarse configs to refine (aggregated across seeds):")
        for m in topk:
            print(
                f"- {m.run_tag}: success={m.success_mean:.3f} fraction_valid={m.fraction_valid_mean:.3f} "
                f"path_best={m.path_length_best_mean:.3f} smooth_best={m.smoothness_best_mean:.3f} "
                f"t_total={m.t_inference_total_mean:.3f} (n={m.n_rows})"
            )

        for coarse in topk:
            diffusion_method, steps, order, use_cont_time = _parse_coarse_tag(coarse.run_tag)

            for (
                dpm_method,
                skip_type,
                solver_type,
                lower_order_final,
                denoise_to_zero,
                prior_weight_with_guide,
                n_guide_steps,
                guide_lr,
                max_perturb_x,
                t_start_guide_steps_fraction,
            ) in _iter_product(
                refine_methods,
                refine_skip_types,
                refine_solver_types,
                refine_lof,
                refine_dz,
                refine_prior_weights,
                refine_n_guide_steps,
                refine_guide_lr,
                refine_max_perturb_x,
                refine_t_start_guide_steps_fraction,
            ):
                cfg = copy.deepcopy(base_cfg)
                if args.planner_alg:
                    cfg["planner_alg"] = args.planner_alg

                cfg["diffusion_sampling_method"] = diffusion_method
                cfg["dpm_solver"]["dpm_solver_steps"] = steps
                cfg["dpm_solver"]["dpm_solver_order"] = order
                cfg["dpm_solver"]["use_continuous_time"] = bool(use_cont_time)
                cfg["dpm_solver"]["dpm_solver_method"] = dpm_method
                cfg["dpm_solver"]["dpm_solver_skip_type"] = skip_type
                cfg["dpm_solver"]["dpm_solver_solver_type"] = solver_type
                cfg["dpm_solver"]["lower_order_final"] = bool(lower_order_final)
                cfg["dpm_solver"]["dpm_solver_denoise_to_zero"] = bool(denoise_to_zero)
                cfg["dpm_solver"]["prior_weight_with_guide"] = float(prior_weight_with_guide)
                cfg["dpm_solver"]["n_guide_steps"] = int(n_guide_steps)
                cfg["dpm_solver"]["guide_lr"] = float(guide_lr)
                cfg["dpm_solver"]["max_perturb_x"] = float(max_perturb_x)
                cfg["dpm_solver"]["t_start_guide_steps_fraction"] = float(t_start_guide_steps_fraction)

                # Keep algorithm_type consistent with the coarse winner.
                cfg["dpm_solver"]["dpm_solver_algorithm_type"] = (
                    "dpmsolver++" if diffusion_method == "dpm_solver_pp" else "dpmsolver"
                )

                run_tag = (
                    f"{coarse.run_tag}_m{dpm_method}_skip{skip_type}_sol{solver_type}"
                    f"_lof{int(lower_order_final)}_dz{int(denoise_to_zero)}"
                )
                if args.refine_prior_weights.strip():
                    run_tag += f"_pw{_format_float_tag(prior_weight_with_guide)}"
                if args.refine_n_guide_steps.strip():
                    run_tag += f"_ng{int(n_guide_steps)}"
                if args.refine_guide_lr.strip():
                    run_tag += f"_glr{_format_float_tag(guide_lr)}"
                if args.refine_max_perturb_x.strip():
                    run_tag += f"_mpx{_format_float_tag(max_perturb_x)}"
                if args.refine_t_start_guide_steps_fraction.strip():
                    run_tag += f"_tsg{_format_float_tag(t_start_guide_steps_fraction)}"

                cfg_name = os.path.splitext(os.path.basename(args.base_cfg))[0]
                cfg_path = os.path.join(args.output_dir, f"{cfg_name}__{run_tag}.yaml")
                _write_yaml(cfg, cfg_path)

                run_inference(run_tag, cfg_path)

        if args.collect_metrics or args.phase == "all":
            collect_metrics()


if __name__ == "__main__":
    main()
