"""
DMEval vNext（L1）命令行入口。

调用链（高度概括）：
CLI -> Hydra compose 配置 -> Stage I tune / Stage II compare -> runner 子进程 -> adapter 抽取 -> 聚合/排名/绘图。

这里刻意不使用 `@hydra.main`：
- 避免 Hydra 接管 cwd/outputs 目录体系（符合 `工具描述文档.md` 的 L1 原则）
- 只使用 Hydra 的 defaults + overrides 组合能力（compose API）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config_loader import load_hydra_config
from .stage_compare import run_compare
from .stage_tune import run_tune
from .util import cfg_to_yaml_str


def _default_config_path() -> Path | None:
    """当用户没传 `--config` 时使用的默认配置路径。"""
    candidate = Path("dmeval/conf/config.yaml")
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    """
    CLI 主函数。

    参数:
      argv: 传入自定义参数列表（测试/嵌入式调用时有用）；None 表示使用 sys.argv。

    返回:
      进程退出码（0 表示成功）。
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
            "Hydra override（可重复），例如："
            "-o pipeline.root=outputs/run1 -o common_inference_args.device=cuda:0"
        ),
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)
    subparsers.add_parser("tune", help="Stage I: tune samplers on a single scenario.")
    subparsers.add_parser("compare", help="Stage II: compare samplers across scenarios using best_configs.")
    subparsers.add_parser("run", help="Run Stage I then Stage II in a single process (recommended).")
    subparsers.add_parser("explain", help="Print the resolved config (after Hydra composition).")

    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else _default_config_path()
    if config_path is None:
        parser.error("Missing --config and default dmeval/conf/config.yaml not found.")
    if not config_path.exists():
        parser.error(f"--config not found: {config_path}")

    # 通过 Hydra compose API 加载配置（只做组合/覆盖，不触发 Hydra 的运行目录管理）。
    cfg = load_hydra_config(config_path=config_path, overrides=list(args.override))

    # 分发子命令。`run` 会按顺序执行 tune -> compare，并共享同一个 pipeline.root（保证闭环一致）。
    if args.cmd == "tune":
        run_tune(cfg)
        return 0
    if args.cmd == "compare":
        run_compare(cfg)
        return 0
    if args.cmd == "run":
        run_tune(cfg)
        run_compare(cfg)
        return 0
    if args.cmd == "explain":
        # 方便你调试 defaults/overrides 生效情况。
        print(cfg_to_yaml_str(cfg))
        return 0

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
