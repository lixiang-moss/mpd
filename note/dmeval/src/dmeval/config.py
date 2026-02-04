from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml


@dataclass(frozen=True)
class JobConfig:
    run_tag: str
    cmd: List[str]
    seeds: List[int]
    workdir: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    timeout_sec: int = 0


@dataclass(frozen=True)
class DMEvalConfig:
    version: int
    results_root: str
    defaults: Dict[str, Any]
    jobs: List[JobConfig]


def _require_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping at {path}, got {type(value).__name__}")
    return value


def load_config(path: Union[str, Path]) -> DMEvalConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}

    root = _require_mapping(data, path=str(path))
    version = int(root.get("version", 1))
    results_root = str(root["results_root"])
    defaults = dict(root.get("defaults", {}) or {})
    jobs_raw = root.get("jobs", [])
    if not isinstance(jobs_raw, list):
        raise ValueError("Expected 'jobs' to be a list.")

    jobs: List[JobConfig] = []
    for idx, job in enumerate(jobs_raw):
        job_m = _require_mapping(job, path=f"{path}:jobs[{idx}]")
        run_tag = str(job_m["run_tag"])
        cmd = job_m.get("cmd", None)
        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise ValueError(f"{path}:jobs[{idx}].cmd must be a list[str].")
        seeds_raw = job_m.get("seeds", [defaults.get("seed", 2)])
        if isinstance(seeds_raw, int):
            seeds = [int(seeds_raw)]
        elif isinstance(seeds_raw, list):
            seeds = [int(s) for s in seeds_raw]
        else:
            raise ValueError(f"{path}:jobs[{idx}].seeds must be int or list[int].")

        workdir = job_m.get("workdir", defaults.get("workdir", None))
        env = job_m.get("env", defaults.get("env", None))
        timeout_sec = int(job_m.get("timeout_sec", defaults.get("timeout_sec", 0)) or 0)

        jobs.append(
            JobConfig(
                run_tag=run_tag,
                cmd=list(cmd),
                seeds=seeds,
                workdir=str(workdir) if workdir is not None else None,
                env=dict(env) if isinstance(env, Mapping) else None,
                timeout_sec=timeout_sec,
            )
        )

    return DMEvalConfig(version=version, results_root=results_root, defaults=defaults, jobs=jobs)
