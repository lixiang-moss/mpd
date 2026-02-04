import argparse
import sys

from dmeval.collect import collect_command
from dmeval.plot import plot_command
from dmeval.compare import compare_command
from dmeval.runner import run_command
from dmeval.tune import tune_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dmeval", description="DMEval: decoupled evaluation framework.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run jobs from a YAML config (serial execution).")
    run_p.add_argument("--config", required=True, help="Path to DMEval YAML config.")
    run_p.add_argument("--dry_run", action="store_true", help="Print commands without executing.")

    collect_p = sub.add_parser("collect", help="Collect metrics from a results tree into CSV.")
    collect_p.add_argument("--adapter", default="mpd", help="Adapter name (default: mpd).")
    collect_p.add_argument("--results_root", required=True, help="Root dir containing run_tag/seed/results*.pt.")
    collect_p.add_argument("--out_dir", required=True, help="Output dir for CSV files.")
    collect_p.add_argument(
        "--include_all_config",
        action="store_true",
        help="Flatten and include all keys from args_inference.yaml (can be wide).",
    )

    plot_p = sub.add_parser("plot", help="Generate basic plots from run_metrics.csv.")
    plot_p.add_argument("--run_metrics", required=True, help="Path to run_metrics.csv (from `dmeval collect`).")
    plot_p.add_argument("--out_dir", required=True, help="Directory to write plots.")
    plot_p.add_argument("--scenario", default="", help="Optional scenario filter (e.g., scene01).")

    tune_p = sub.add_parser("tune", help="Stage I: sampler hyperparameter tuning (serial grid search).")
    tune_p.add_argument("--config", required=True, help="Path to DMEval tuning YAML config.")
    tune_p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable), e.g. scenario=spheres3d or common_inference_args.device=cuda:0",
    )

    compare_p = sub.add_parser("compare", help="Stage II: compare samplers using tuned configs across scenarios.")
    compare_p.add_argument("--config", required=True, help="Path to DMEval compare YAML config.")
    compare_p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable), e.g. compare.seeds=[2,3] or common_inference_args.n_start_goal_states=100",
    )

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        return run_command(config_path=args.config, dry_run=args.dry_run)
    if args.command == "collect":
        return collect_command(
            adapter=args.adapter,
            results_root=args.results_root,
            out_dir=args.out_dir,
            include_all_config=args.include_all_config,
        )
    if args.command == "plot":
        return plot_command(run_metrics=args.run_metrics, out_dir=args.out_dir, scenario=args.scenario)
    if args.command == "tune":
        return tune_command(config_path=args.config, overrides=args.override)
    if args.command == "compare":
        return compare_command(config_path=args.config, overrides=args.override)

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
