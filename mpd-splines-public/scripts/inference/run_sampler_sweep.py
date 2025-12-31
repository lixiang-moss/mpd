import argparse
import copy
import itertools
import os
import subprocess
from typing import List

import yaml


def _parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _time_modes_from_arg(mode: str) -> List[bool]:
    if mode == "both":
        return [False, True]
    if mode == "continuous":
        return [True]
    if mode == "discrete":
        return [False]
    raise ValueError(f"Unsupported time mode: {mode}")


def _run_command(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_cfg",
        required=True,
        help="Path to base inference YAML config.",
    )
    parser.add_argument(
        "--method",
        default="unipc",
        choices=["unipc", "dpm_solver"],
        help="Sampler method to sweep.",
    )
    parser.add_argument(
        "--planner_alg",
        default="",
        help="Override planner_alg in the generated configs (e.g., mpd).",
    )
    parser.add_argument(
        "--steps",
        default="20,40",
        help="Comma-separated steps list, e.g., 20,40",
    )
    parser.add_argument(
        "--orders",
        default="1,2,3",
        help="Comma-separated order list, e.g., 1,2,3",
    )
    parser.add_argument(
        "--time_modes",
        default="both",
        choices=["continuous", "discrete", "both"],
        help="Evaluate continuous time, discrete time, or both.",
    )
    parser.add_argument(
        "--variants",
        default="bh1",
        help="Comma-separated UniPC variants to sweep (only for method=unipc).",
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        help="Enable clip_denoised for UniPC runs.",
    )
    parser.add_argument(
        "--output_dir",
        default="mpd-splines-public/scripts/inference/cfgs/generated",
        help="Directory to write generated configs.",
    )
    parser.add_argument(
        "--results_root",
        default="mpd-splines-public/scripts/inference/logs/sweep",
        help="Root directory for run results.",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generate_only", action="store_true", help="Only generate configs, do not run.")
    parser.add_argument(
        "--collect_metrics",
        action="store_true",
        help="Collect metrics into CSV after the sweep finishes.",
    )
    args = parser.parse_args()

    steps_list = _parse_int_list(args.steps)
    orders_list = _parse_int_list(args.orders)
    variants_list = _parse_str_list(args.variants) if args.method == "unipc" else ["default"]
    time_modes = _time_modes_from_arg(args.time_modes)

    if not steps_list or not orders_list:
        raise ValueError("Both --steps and --orders must contain at least one value.")

    with open(args.base_cfg, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_root, exist_ok=True)

    combos = itertools.product(steps_list, orders_list, time_modes, variants_list)
    for steps, order, use_cont_time, variant in combos:
        cfg = copy.deepcopy(base_cfg)
        cfg["diffusion_sampling_method"] = args.method
        if args.planner_alg:
            cfg["planner_alg"] = args.planner_alg

        run_tag = f"{args.method}_steps{steps}_order{order}_ct{int(use_cont_time)}"

        if args.method == "unipc":
            if "unipc" not in cfg:
                raise ValueError("Base config missing 'unipc' section.")
            cfg["unipc"]["unipc_steps"] = steps
            cfg["unipc"]["unipc_order"] = order
            cfg["unipc"]["use_continuous_time"] = use_cont_time
            cfg["unipc"]["unipc_variant"] = variant
            cfg["unipc"]["clip_denoised"] = bool(args.clip_denoised)
            run_tag += f"_var{variant}"
        else:
            if "dpm_solver" not in cfg:
                raise ValueError("Base config missing 'dpm_solver' section.")
            cfg["dpm_solver"]["dpm_solver_steps"] = steps
            cfg["dpm_solver"]["dpm_solver_order"] = order
            cfg["dpm_solver"]["use_continuous_time"] = use_cont_time

        cfg_name = os.path.splitext(os.path.basename(args.base_cfg))[0]
        out_cfg = os.path.join(args.output_dir, f"{cfg_name}__{run_tag}.yaml")
        with open(out_cfg, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        if args.generate_only:
            print(f"Generated config: {out_cfg}")
            continue

        results_dir = os.path.join(args.results_root, run_tag)
        cmd = [
            "python3",
            "mpd-splines-public/scripts/inference/inference.py",
            "--cfg_inference_path",
            out_cfg,
            "--results_dir",
            results_dir,
            "--seed",
            str(args.seed),
            "--save_results_single_plan_low_mem",
            "true",
            "--device",
            args.device,
        ]
        _run_command(cmd)

    if args.collect_metrics and not args.generate_only:
        collect_cmd = [
            "python3",
            "mpd-splines-public/scripts/inference/collect_sweep_metrics.py",
            "--results_root",
            args.results_root,
        ]
        _run_command(collect_cmd)


if __name__ == "__main__":
    main()
