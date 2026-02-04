"""
通用工具函数集合（与具体 planner/adapter 无关）。

这里放的是“稳定的基础设施”能力：
- 目录创建/清理（保证 pipeline.root 可控且可追溯）
- YAML/JSONL 读写
- patch 合并（base_cfg + best_patch 的深度合并）
- dotpath 转嵌套 dict（用于从 search space 生成 patch）
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from omegaconf import DictConfig, OmegaConf


def now_utc_iso() -> str:
    """返回 UTC 时间的 ISO 字符串（用于 manifest 记录）。"""
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    """确保目录存在（mkdir -p）。"""
    path.mkdir(parents=True, exist_ok=True)


def ensure_empty_dir(path: Path, *, allow_overwrite: bool) -> None:
    """
    创建一个“干净”的目录。

    - 如果目录已存在：allow_overwrite=false 时直接报错（防止覆盖实验结果）
    - allow_overwrite=true 时会先删除再创建（适合 CI/单测/快速迭代）
    """
    if path.exists():
        if not allow_overwrite:
            raise FileExistsError(f"Refuse to overwrite existing directory: {path}")
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def safe_relpath(path: Path) -> str:
    """尽量输出相对路径（用于日志/manifest），失败则回退绝对路径。"""
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML，并强制要求根节点为 mapping（dict）。"""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got {type(data)}")
    return data


def write_yaml(path: Path, data: Any) -> None:
    """写 YAML（保持 key 顺序；允许中文）。"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_text(path: Path, text: str) -> None:
    """写文本文件（UTF-8）。"""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """写 jsonl（每行一个 JSON dict）。"""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """
    深度合并 dict（就地修改 base）。

    约定：
    - 若 base[k] 与 patch[k] 都是 dict：递归合并
    - 否则 patch[k] 覆盖 base[k]

    这个逻辑用于把 `best_patch.yaml` 叠加到 `base_cfg` 上生成最终 cfg。
    """
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


_DOTPATH_TOKEN = re.compile(r"^[A-Za-z0-9_]+$")


def nested_dict_from_dotpath(dotpath: str, value: Any) -> dict[str, Any]:
    """
    把形如 `dpm_solver.dpm_solver_steps` 的 dotpath 转为嵌套 dict：

    - 输入: ("a.b.c", 1)
    - 输出: {"a": {"b": {"c": 1}}}

    用途：search space 里用 dotpath 表示要 patch 的字段，生成 patch.yaml。
    """
    parts = [p for p in dotpath.split(".") if p]
    if not parts:
        raise ValueError("Empty dotpath")
    for p in parts:
        if not _DOTPATH_TOKEN.match(p):
            raise ValueError(f"Unsupported dotpath token: {p!r} in {dotpath!r}")
    d: dict[str, Any] = {}
    cur = d
    for p in parts[:-1]:
        nxt: dict[str, Any] = {}
        cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return d


def merge_patches(*patches: dict[str, Any]) -> dict[str, Any]:
    """把多个 patch dict 依次深度合并成一个新的 dict。"""
    out: dict[str, Any] = {}
    for p in patches:
        deep_update(out, p)
    return out


def cfg_to_yaml_str(cfg: DictConfig) -> str:
    """把 OmegaConf/DictConfig 转为可打印的 YAML（用于 `dmeval explain`）。"""
    return OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False)


def maybe_expand_envvars(value: str) -> str:
    """展开环境变量（例如 ${MPDLX_DATA_ROOT}），主要给 planner cfg 使用。"""
    return os.path.expandvars(value)


@dataclass(frozen=True)
class RunManifest:
    """
    运行元信息（当前实现里主要写入 YAML manifest）。

    注：为了保持 L1 简洁，这里没有做复杂的 schema/version 管理；
    你如果后续要扩展为更严谨的 manifest，可在此处集中演进。
    """
    tool: str
    started_at_utc: str
    config_path: str
    overrides: list[str]
