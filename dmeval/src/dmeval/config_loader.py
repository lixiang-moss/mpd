"""
Hydra 配置加载器（compose API）。

关键原则（对应 `工具描述文档.md` 的 L1 约束）：
- Hydra 只用于配置组合（defaults + overrides）
- 不使用 `@hydra.main`，从而避免 Hydra 接管 cwd/outputs/launcher
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_hydra_config(*, config_path: Path, overrides: list[str]) -> DictConfig:
    """
    加载并 compose 一个 Hydra 配置。

    参数:
      config_path: 入口 YAML（例如 dmeval/conf/config.yaml）
      overrides: Hydra overrides（例如 ["planner=mpd", "common_inference_args.device=cuda:0"]）

    返回:
      DictConfig（已完成 defaults 合并与 overrides 覆盖）。
    """
    config_path = config_path.resolve()
    config_dir = str(config_path.parent)
    config_name = config_path.stem

    # initialize_config_dir 只用于告诉 Hydra 去哪里找配置文件，不会修改 cwd。
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
