"""
DMEval vNext (L1) command-line entrypoint.

Call chain (high level):
CLI -> Hydra config composition -> Stage I tune / Stage II compare -> runner subprocess -> adapter extraction -> aggregation/ranking/plotting.

We intentionally avoid `@hydra.main`:
- Prevent Hydra from taking over cwd/outputs directory management (L1 principle)
- Use only Hydra's composition features (defaults + overrides via compose API)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config_loader import load_hydra_config
from .stage_compare import run_compare
from .stage_tune import run_tune
from .util import cfg_to_yaml_str

_LINE = "-" * 72


def _log(msg: str) -> None:
    """Lightweight terminal logging (enabled by default)."""
    print(f"[dmeval] {msg}", flush=True)


def _section(title: str) -> None:
    """Print a section divider for readability."""
    print(_LINE, flush=True)
    _log(title)
    print(_LINE, flush=True)


def _default_config_path() -> Path | None:
    """Default config path when the user does not provide `--config`."""
    candidate = Path("dmeval/conf/config.yaml")
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    """
    CLI main function.

    Args:
      argv: optional custom argv list (useful for tests/embedded runs). None uses `sys.argv`.

    Returns:
      Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="dmeval", description="DMEval vNext (L1) CLI")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a Hydra YAML config (e.g. dmeval/conf/config.yaml).",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help=(
            "Hydra override (repeatable), e.g.: "
            "-o pipeline.root=outputs/run1 -o common_inference_args.device=cuda:0"
        ),
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("tune", help="Stage I: tune samplers on a single scenario.")
    subparsers.add_parser("compare", help="Stage II: compare samplers across scenarios using best_configs.")
    subparsers.add_parser("run", help="Run Stage I then Stage II in a single process (recommended).")
    subparsers.add_parser("explain", help="Print the resolved config (after Hydra composition).")

    args = parser.parse_args(argv)
    started = time.monotonic()

    config_path = Path(args.config) if args.config else _default_config_path()
    if config_path is None:
        parser.error("Missing --config and default dmeval/conf/config.yaml not found.")
    if not config_path.exists():
        parser.error(f"--config not found: {config_path}")

    # Load config via Hydra compose API (composition/overrides only; no Hydra run-dir management).
    cfg = load_hydra_config(config_path=config_path, overrides=list(args.override))
    _section(f"COMMAND START | {args.cmd}")
    _log(f"config={config_path}")
    _log(f"pipeline.root={cfg.pipeline.root}")
    _log(f"overrides={len(args.override)}")

    # Dispatch subcommands. `run` executes tune -> compare and shares the same pipeline.root.
    if args.cmd == "tune":
        run_tune(cfg)
        _section("COMMAND END | tune")
        _log(f"completed in {time.monotonic() - started:.1f}s")
        return 0
    if args.cmd == "compare":
        run_compare(cfg)
        _section("COMMAND END | compare")
        _log(f"completed in {time.monotonic() - started:.1f}s")
        return 0
    if args.cmd == "run":
        _section("TRANSITION | STAGE I (tune)")
        run_tune(cfg)
        _section("TRANSITION | STAGE II (compare)")
        run_compare(cfg)
        _section("COMMAND END | run")
        _log(f"completed in {time.monotonic() - started:.1f}s")
        return 0
    if args.cmd == "explain":
        # Helpful for debugging Hydra defaults/overrides.
        print(cfg_to_yaml_str(cfg))
        _section("COMMAND END | explain")
        _log(f"completed in {time.monotonic() - started:.1f}s")
        return 0

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
