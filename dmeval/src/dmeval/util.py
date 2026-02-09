"""
General utility helpers (planner/adapter agnostic).

This module provides "stable infrastructure" helpers:
- Directory creation/cleanup (keeping `pipeline.root` controlled and traceable)
- YAML/JSONL read/write
- Patch merging (deep merge of `base_cfg` + `best_patch`)
- Dotpath -> nested dict conversion (to build patches from a search space)
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
    """Return the current UTC time as an ISO string (for manifests)."""
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists (mkdir -p)."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_empty_dir(path: Path, *, allow_overwrite: bool) -> None:
    """
    Create a "clean" directory.

    - If the directory exists: raise when `allow_overwrite=false` (prevents clobbering results)
    - When `allow_overwrite=true`: delete first then create (useful for CI/tests/iteration)
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
    """Prefer relative paths for logs/manifests; fall back to absolute on failure."""
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and require the root node to be a mapping (dict)."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got {type(data)}")
    return data


def write_yaml(path: Path, data: Any) -> None:
    """Write YAML (preserve key order; allow Unicode)."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file."""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL (one JSON dict per line)."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge dicts (mutates `base` in place).

    Rules:
    - If both `base[k]` and `patch[k]` are dicts: merge recursively
    - Otherwise: `patch[k]` overwrites `base[k]`

    This is used to overlay `best_patch.yaml` onto `base_cfg` to produce the final config.
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
    Convert a dotpath like `dpm_solver.dpm_solver_steps` into a nested dict:

    - Input: ("a.b.c", 1)
    - Output: {"a": {"b": {"c": 1}}}

    Used when the search space uses dotpaths to specify which fields to patch.
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
    """Deep-merge multiple patch dicts (in order) into a new dict."""
    out: dict[str, Any] = {}
    for p in patches:
        deep_update(out, p)
    return out


def cfg_to_yaml_str(cfg: DictConfig) -> str:
    """Convert an OmegaConf/DictConfig to printable YAML (used by `dmeval explain`)."""
    return OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False)


def maybe_expand_envvars(value: str) -> str:
    """Expand environment variables (e.g. `${MPDLX_DATA_ROOT}`), mainly for planner configs."""
    return os.path.expandvars(value)


@dataclass(frozen=True)
class RunManifest:
    """
    Run metadata (currently written to YAML manifests).

    Note: to keep the L1 build minimal, we do not implement schema/version management here.
    If you later want a stricter manifest format, evolve it centrally in this class.
    """
    tool: str
    started_at_utc: str
    config_path: str
    overrides: list[str]
