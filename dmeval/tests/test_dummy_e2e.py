"""
Minimal end-to-end unit test (dummy planner).

Goal:
- Run the full DMEval loop with the MPD-free dummy planner:
  run_tune -> best_configs -> run_compare -> generate CSVs and plots

Notes:
- This test does not validate numeric correctness (dummy metrics are a toy model).
- It validates that the call chain and artifact contract remain stable across refactors.
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dmeval" / "src"))
# Avoid interference from any locally installed package with the same name (or egg-info).
sys.modules.pop("dmeval", None)

from dmeval.config_loader import load_hydra_config  # noqa: E402
from dmeval.stage_compare import run_compare  # noqa: E402
from dmeval.stage_tune import run_tune  # noqa: E402


class DummyE2ETest(unittest.TestCase):
    def test_run_end_to_end_with_dummy_planner(self) -> None:
        # Test output directory: clear it before each run to keep the test reproducible.
        out_root = REPO_ROOT / ".tmp" / "dmeval_unittest"
        if out_root.exists():
            shutil.rmtree(out_root)

        # Use Hydra compose to load a minimal config and override pipeline.root.
        cfg = load_hydra_config(
            config_path=REPO_ROOT / "dmeval" / "conf" / "config_dummy_small.yaml",
            overrides=[
                f"pipeline.root={out_root.as_posix()}",
                "pipeline.allow_overwrite=true",
            ],
        )

        # Stage I: should write manifest and best_patch.yaml
        run_tune(cfg)
        self.assertTrue((out_root / "tune_manifest.yaml").exists())
        self.assertTrue((out_root / "best_configs" / "ddim" / "best_patch.yaml").exists())

        # Stage II: should write run_metrics.csv and at least one plot
        run_compare(cfg)
        self.assertTrue((out_root / "compare" / "run_metrics.csv").exists())
        self.assertTrue((out_root / "compare" / "plots" / "planar2link__success.png").exists())


if __name__ == "__main__":
    unittest.main()
