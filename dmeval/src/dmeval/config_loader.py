"""
Hydra config loader (compose API).

Key principles (L1 constraints):
- Use Hydra only for composition (defaults + overrides)
- Do not use `@hydra.main`, so Hydra does not take over cwd/outputs/launcher
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_hydra_config(*, config_path: Path, overrides: list[str]) -> DictConfig:
    """
    Load and compose a Hydra config.

    Args:
      config_path: entry YAML (e.g. `dmeval/conf/config.yaml`)
      overrides: Hydra overrides (e.g. `["planner=mpd", "common_inference_args.device=cuda:0"]`)

    Returns:
      A `DictConfig` after applying defaults and overrides.
    """
    config_path = config_path.resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    # `initialize_config_dir` only tells Hydra where to find configs; it does not change cwd.
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
