#!/usr/bin/env bash
set -euo pipefail

# Run an MPD sweep (sequential) and collect metrics into CSV files.
# Adjust BASE_CFG and DEVICE as needed.

BASE_CFG="mpd-splines-public/scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml"
DEVICE="cuda:0"

python3 mpd-splines-public/scripts/inference/run_sampler_sweep.py \
  --base_cfg "${BASE_CFG}" \
  --method dpm_solver \
  --planner_alg mpd \
  --steps 15,20,26,32,40 \
  --orders 2 \
  --time_modes both \
  --results_root mpd-splines-public/scripts/inference/logs/sweep \
  --device "${DEVICE}" \
  --collect_metrics
