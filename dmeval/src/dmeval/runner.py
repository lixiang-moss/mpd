"""
Subprocess runner (serial execution).

Responsibilities:
- Convert "planner entrypoint + inference args" into a unified CLI invocation
- Run the system-under-test (e.g., MPD) via `subprocess.run` (serially)
- Write stdout/stderr to a log file for reproducibility and debugging

Notes:
- No parallelism/queueing here (L1 principle: serial only)
- This module does not parse results; adapters handle parsing (separation of concerns)
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .util import ensure_dir, safe_relpath, write_text


def _bool_to_cli(value: bool) -> str:
    """Convert Python bool to common CLI-style `true`/`false` strings."""
    return "true" if value else "false"


def _format_cli_value(value: Any) -> str:
    """Format a parameter value as a CLI string (currently mainly handles bool)."""
    if isinstance(value, bool):
        return _bool_to_cli(value)
    return str(value)


def run_planner_subprocess(
    *,
    python: str,
    entrypoint: str,
    workdir: Path,
    cfg_inference_path: Path,
    results_dir: Path,
    seed: int,
    selection_start_goal: str,
    n_start_goal_states: int,
    save_results_single_plan_low_mem: bool,
    device: str,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    """
    Run one planner subprocess serially (in the spec: one run = candidate × seed).

    Conventions:
    - `results_dir` is chosen by DMEval; the planner may create a seed subdirectory inside it
    - DMEval writes `dmeval_cmd.txt` under `results_dir` to record the exact command (traceability)
    - Subprocess stdout/stderr are written to `log_path` (default: `dmeval_subprocess.log`)
    """
    ensure_dir(results_dir)

    # Unified CLI parameters (this corresponds to the example command in the spec).
    cmd: list[str] = [
        python,
        entrypoint,
        "--cfg_inference_path",
        str(cfg_inference_path),
        "--results_dir",
        str(results_dir),
        "--seed",
        str(seed),
        "--selection_start_goal",
        selection_start_goal,
        "--n_start_goal_states",
        str(n_start_goal_states),
        "--save_results_single_plan_low_mem",
        _format_cli_value(save_results_single_plan_low_mem),
        "--device",
        device,
    ]
    if extra_args:
        cmd.extend(extra_args)

    ensure_dir(results_dir)
    # Save a copy-pastable final command for repro/debugging.
    write_text(results_dir / "dmeval_cmd.txt", shlex.join(cmd) + "\n")

    # Inherit current environment variables (e.g., data roots, CUDA); optionally override via `env`.
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    if log_path is None:
        log_path = results_dir / "dmeval_subprocess.log"

    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as log_f:
        # check=False: we want the log file even on failure, then raise a single structured error.
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=run_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        rel_log_path = safe_relpath(log_path)
        rel_workdir = safe_relpath(workdir)
        raise RuntimeError(
            f"Planner subprocess failed (exit={proc.returncode}, seed={seed}, cwd={rel_workdir}). See log: {rel_log_path}"
        )
