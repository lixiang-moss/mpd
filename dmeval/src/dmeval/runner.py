"""
子进程 Runner（串行执行）。

职责：
- 把“planner 入口 + 推理参数”统一转成命令行参数
- 串行 `subprocess.run` 运行被测系统（例如 MPD）
- 把 stdout/stderr 写到 log 文件，便于复现与排障

注意：
- 这里不做并行/队列（符合 L1 “只串行”原则）
- 这里不解析结果，解析由 Adapter 负责（解耦）
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .util import ensure_dir, write_text


def _bool_to_cli(value: bool) -> str:
    """把 Python bool 转成许多 CLI 常见的 true/false 字符串。"""
    return "true" if value else "false"


def _format_cli_value(value: Any) -> str:
    """将参数值格式化为 CLI 字符串（目前主要处理 bool）。"""
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
    串行运行一次 planner 子进程（对应规格里的“一次 run = candidate × seed”）。

    约定：
    - results_dir 由 DMEval 指定；planner 自己决定是否创建 seed 子目录
    - DMEval 会在 results_dir 下写 `dmeval_cmd.txt`，记录最终命令行（可追溯）
    - 子进程 stdout/stderr 会写到 log_path（默认 dmeval_subprocess.log）
    """
    ensure_dir(results_dir)

    # 统一命令行参数：这部分就是 `工具描述文档.md` 里示例命令的实现化。
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
    # 保存一份“可复制粘贴”的最终命令，方便复现与排障。
    write_text(results_dir / "dmeval_cmd.txt", shlex.join(cmd) + "\n")

    # 继承当前进程环境变量（例如数据根目录、CUDA 等），必要时可通过 env 追加覆盖。
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    if log_path is None:
        log_path = results_dir / "dmeval_subprocess.log"

    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as log_f:
        # check=False：我们希望在失败时也能先拿到 log 文件，再统一抛异常。
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=run_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Planner subprocess failed (exit={proc.returncode}). See log: {log_path}")
