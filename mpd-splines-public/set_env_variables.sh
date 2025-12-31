#!/usr/bin/env bash

# Auto-detect the active env; fallback to the default path.
ENV_PREFIX="${CONDA_PREFIX:-${VIRTUAL_ENV:-${MPD_ENV_PREFIX:-$HOME/miniconda3/envs/mpd-splines-public}}}"
ENV_PREFIX="${ENV_PREFIX%/}"

if [ ! -d "${ENV_PREFIX}" ]; then
  echo "WARNING: Environment path ${ENV_PREFIX} not found. Activate the env first." >&2
  return 0 2>/dev/null || exit 0
fi

export PATH="${ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPATH="${ENV_PREFIX}/include${CPATH:+:${CPATH}}"
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
