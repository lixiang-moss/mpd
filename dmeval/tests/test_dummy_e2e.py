"""
最小端到端单测（dummy planner）。

目标：
- 用不依赖 MPD 的 dummy planner 跑通 DMEval 闭环：
  run_tune -> best_configs -> run_compare -> 生成 CSV 与 plots

说明：
- 这个测试不验证数值正确性（dummy 指标本身是 toy model）
- 只验证“调用链 + 产物契约”是否稳定（便于后续重构不破坏闭环）
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dmeval" / "src"))
# 避免本地可能存在的同名已安装包（或 egg-info）干扰导入，确保测试使用仓库源码。
sys.modules.pop("dmeval", None)

from dmeval.config_loader import load_hydra_config  # noqa: E402
from dmeval.stage_compare import run_compare  # noqa: E402
from dmeval.stage_tune import run_tune  # noqa: E402


class DummyE2ETest(unittest.TestCase):
    def test_run_end_to_end_with_dummy_planner(self) -> None:
        # 单测输出目录：每次运行前清空，确保测试可重复。
        out_root = REPO_ROOT / ".tmp" / "dmeval_unittest"
        if out_root.exists():
            shutil.rmtree(out_root)

        # 直接用 Hydra compose 读取一个最小配置，并覆盖 pipeline.root。
        cfg = load_hydra_config(
            config_path=REPO_ROOT / "dmeval" / "conf" / "config_dummy_small.yaml",
            overrides=[
                f"pipeline.root={out_root.as_posix()}",
                "pipeline.allow_overwrite=true",
            ],
        )

        # Stage I：应写出 manifest 与 best_patch.yaml
        run_tune(cfg)
        self.assertTrue((out_root / "tune_manifest.yaml").exists())
        self.assertTrue((out_root / "best_configs" / "ddim" / "best_patch.yaml").exists())

        # Stage II：应写出 run_metrics.csv 与至少一张 plot
        run_compare(cfg)
        self.assertTrue((out_root / "compare" / "run_metrics.csv").exists())
        self.assertTrue((out_root / "compare" / "plots" / "planar2link__success.png").exists())


if __name__ == "__main__":
    unittest.main()
