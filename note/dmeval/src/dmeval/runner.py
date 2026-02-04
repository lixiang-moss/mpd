from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from dmeval.config import DMEvalConfig, JobConfig, load_config
from dmeval.utils import format_template


@dataclass(frozen=True)
class RunRecord:
    run_tag: str
    seed: int
    cmd: List[str]
    cwd: str
    returncode: int
    elapsed_sec: float
    stdout_path: str
    stderr_path: str


def _merge_env(overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
    env = dict(os.environ)
    if overrides:
        env.update({str(k): str(v) for k, v in overrides.items()})
    return env


def _run_one(job: JobConfig, *, results_root: str, seed: int, dry_run: bool) -> RunRecord:
    cwd = job.workdir or os.getcwd()
    base_out_dir = Path(results_root) / job.run_tag / str(seed)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    variables = {"results_root": results_root, "run_tag": job.run_tag, "seed": str(seed)}
    cmd = [format_template(tok, variables) for tok in job.cmd]

    stdout_path = str(base_out_dir / "dmeval_stdout.txt")
    stderr_path = str(base_out_dir / "dmeval_stderr.txt")

    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return RunRecord(
            run_tag=job.run_tag,
            seed=seed,
            cmd=cmd,
            cwd=cwd,
            returncode=0,
            elapsed_sec=0.0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    start = time.time()
    with open(stdout_path, "w") as stdout_f, open(stderr_path, "w") as stderr_f:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=_merge_env(job.env),
            stdout=stdout_f,
            stderr=stderr_f,
            timeout=job.timeout_sec if job.timeout_sec > 0 else None,
            check=False,
        )
    elapsed = time.time() - start

    return RunRecord(
        run_tag=job.run_tag,
        seed=seed,
        cmd=cmd,
        cwd=cwd,
        returncode=int(proc.returncode),
        elapsed_sec=float(elapsed),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def run_jobs_serial(cfg: DMEvalConfig, *, dry_run: bool) -> List[RunRecord]:
    results_root = format_template(cfg.results_root, {"results_root": cfg.results_root})
    if not Path(results_root).is_absolute():
        base_dir = cfg.defaults.get("workdir", None) or os.getcwd()
        results_root = str(Path(base_dir) / results_root)
    Path(results_root).mkdir(parents=True, exist_ok=True)

    records: List[RunRecord] = []
    for job in cfg.jobs:
        for seed in job.seeds:
            rec = _run_one(job, results_root=results_root, seed=seed, dry_run=dry_run)
            records.append(rec)
            if rec.returncode != 0:
                print(f"[dmeval] WARNING: job failed run_tag={rec.run_tag} seed={rec.seed} rc={rec.returncode}")
    return records


def run_command(*, config_path: str, dry_run: bool) -> int:
    cfg = load_config(config_path)
    records = run_jobs_serial(cfg, dry_run=dry_run)

    results_root = cfg.results_root
    if not Path(results_root).is_absolute():
        base_dir = cfg.defaults.get("workdir", None) or os.getcwd()
        results_root = str(Path(base_dir) / results_root)

    out_dir = Path(results_root) / "_dmeval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_records.yaml"
    out_path.write_text(yaml.safe_dump([asdict(r) for r in records], sort_keys=False))
    print(f"[dmeval] Wrote {out_path}")

    failed = [r for r in records if r.returncode != 0]
    return 1 if failed else 0
