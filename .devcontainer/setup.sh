#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[devcontainer] $*"
}

warn() {
  echo "[devcontainer][warn] $*" >&2
}

MPDLX_ROOT=${MPDLX_ROOT:-/workspaces/MPDLX-B-new}
MPDLX_REPO=${MPDLX_REPO:-${MPDLX_ROOT}/mpd-splines-public}
MPDLX_DATA_ROOT=${MPDLX_DATA_ROOT:-${MPDLX_ROOT}/data_public}
MPDLX_ISAAC_ROOT=${MPDLX_ISAAC_ROOT:-${MPDLX_REPO}/deps/isaacgym}
MPDLX_ISAAC_ARCHIVE=${MPDLX_ISAAC_ARCHIVE:-${MPDLX_ROOT}/IsaacGym_Preview_4_Package.tar.gz}
MPDLX_BUILD_OMPL=${MPDLX_BUILD_OMPL:-1}
MPDLX_ENV_NAME=${MPDLX_ENV_NAME:-mpd-splines-public}
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/opt/conda}

if [ ! -d "${MPDLX_REPO}" ]; then
  log "Repository not found at ${MPDLX_REPO}"
  exit 1
fi

if ! command -v micromamba >/dev/null 2>&1; then
  log "micromamba not available on PATH"
  exit 1
fi

log "Using repository: ${MPDLX_REPO}"
log "Data root: ${MPDLX_DATA_ROOT}"
log "Isaac root: ${MPDLX_ISAAC_ROOT}"

eval "$(micromamba shell hook -s bash)"
export PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"

ensure_safe_git() {
  git config --global --add safe.directory "${MPDLX_REPO}" || true
  git config --global --add safe.directory "${MPDLX_ROOT}" || true
}

ensure_submodules() {
  log "Updating git submodules"
  local submodules=(deps/experiment_launcher deps/pybullet_ompl deps/theseus)

  if [ -d "${MPDLX_REPO}/.git" ]; then
    if ! git -C "${MPDLX_REPO}" submodule update --init --recursive --progress; then
      warn "Submodule update failed; ensure network access or clone with --recurse-submodules."
    fi
    return
  fi

  local missing=()
  for path in "${submodules[@]}"; do
    if [ ! -d "${MPDLX_REPO}/${path}" ] || [ -z "$(ls -A "${MPDLX_REPO}/${path}" 2>/dev/null || true)" ]; then
      missing+=("${path}")
    fi
  done

  if [ ${#missing[@]} -gt 0 ]; then
    warn "Submodule directories missing: ${missing[*]}. Place them manually or use a full clone."
  else
    log "Submodule content present; skipping update (no .git directory)."
  fi
}

ensure_isaac() {
  if [ -d "${MPDLX_ISAAC_ROOT}" ] || [ -L "${MPDLX_ISAAC_ROOT}" ]; then
    log "Isaac Gym already present at ${MPDLX_ISAAC_ROOT}"
    return
  fi

  if [ ! -f "${MPDLX_ISAAC_ARCHIVE}" ]; then
    log "Isaac Gym archive not found at ${MPDLX_ISAAC_ARCHIVE}; set MPDLX_ISAAC_ROOT or provide the archive."
    return
  fi

  log "Extracting Isaac Gym from ${MPDLX_ISAAC_ARCHIVE}"
  mkdir -p "$(dirname "${MPDLX_ISAAC_ROOT}")"
  if ! tar -xf "${MPDLX_ISAAC_ARCHIVE}" -C "$(dirname "${MPDLX_ISAAC_ROOT}")"; then
    warn "Failed to extract Isaac Gym from ${MPDLX_ISAAC_ARCHIVE}; install it manually or set MPDLX_ISAAC_ROOT."
    return
  fi

  # Try to locate the extracted folder if the archive created a nested directory.
  if [ ! -d "${MPDLX_ISAAC_ROOT}" ]; then
    nested_dir="$(find "$(dirname "${MPDLX_ISAAC_ROOT}")" -maxdepth 2 -type d -name 'isaacgym' | head -n 1 || true)"
    if [ -n "${nested_dir}" ]; then
      ln -s "${nested_dir}" "${MPDLX_ISAAC_ROOT}"
      log "Linked Isaac Gym from ${nested_dir}"
    fi
  fi

  if [ ! -d "${MPDLX_ISAAC_ROOT}" ]; then
    warn "Failed to place Isaac Gym under ${MPDLX_ISAAC_ROOT} (set MPDLX_ISAAC_ROOT or extract manually)."
  fi
}

link_datasets() {
  for name in data_trajectories data_trained_models; do
    target="${MPDLX_DATA_ROOT}/${name}"
    link="${MPDLX_REPO}/${name}"
    if [ -L "${link}" ]; then
      # Refresh broken symlinks to the new workspace path.
      if [ ! -e "${link}" ]; then
        rm "${link}"
      else
        continue
      fi
    elif [ -e "${link}" ]; then
      continue
    fi
    if [ -d "${target}" ]; then
      ln -s "${target}" "${link}"
      log "Linked ${name} -> ${target}"
    else
      log "Skipping link for ${name}; missing target ${target}"
    fi
  done
}

ensure_env() {
  log "Creating micromamba environment ${MPDLX_ENV_NAME} if needed"
  pushd "${MPDLX_REPO}" >/dev/null
  if ! micromamba env list | awk '{print $1}' | grep -qx "${MPDLX_ENV_NAME}"; then
    micromamba env create -y -f environment.yml -n "${MPDLX_ENV_NAME}"
  fi
  micromamba config set always_yes true
  # Some conda activate scripts (e.g., binutils) expect unset vars; relax -u while activating.
  set +u
  micromamba activate "${MPDLX_ENV_NAME}"
  set -u

  # Reinstall editable packages to ensure consistency when the workspace is bind-mounted.
  pip install -e ./deps/experiment_launcher
  pip install -e ./deps/isaacgym/python
  pip install -e ./deps/theseus/torchkin
  pip install -e ./mpd/torch_robotics
  pip install -e ./mpd/motion_planning_baselines
  pip install -e .
  popd >/dev/null
}

configure_shell_hook() {
  shell_rc="${HOME}/.bashrc"
  if ! grep -q "micromamba shell hook" "${shell_rc}"; then
    echo 'eval "$(micromamba shell hook -s bash)"' >>"${shell_rc}"
  fi
  if ! grep -q "micromamba activate ${MPDLX_ENV_NAME}" "${shell_rc}"; then
    echo "micromamba activate ${MPDLX_ENV_NAME}" >>"${shell_rc}"
  fi
}

build_pybullet_ompl() {
  if [ "${MPDLX_BUILD_OMPL}" = "0" ]; then
    log "Skipping pybullet_ompl build (MPDLX_BUILD_OMPL=0)"
    return
  fi

  log "Building pybullet_ompl and OMPL bindings"
  pushd "${MPDLX_REPO}/deps/pybullet_ompl" >/dev/null

  if [ ! -d ompl ]; then
    if ! git clone https://github.com/ompl/ompl.git; then
      warn "git clone ompl failed; skipping pybullet_ompl build."
      popd >/dev/null
      return
    fi
  fi

  pushd ompl >/dev/null
  git fetch --tags || warn "ompl fetch failed (continuing with existing checkout)."
  git checkout fca10b4bd4840856c7a9f50d1ee2688ba77e25aa || warn "ompl checkout failed; using current branch."
  mkdir -p build/Release
  pushd build/Release >/dev/null

  PYTHON_EXEC="$(micromamba run -n "${MPDLX_ENV_NAME}" which python)"
  if ! micromamba run -n "${MPDLX_ENV_NAME}" cmake -DCMAKE_DISABLE_FIND_PACKAGE_pypy=ON ../.. -DPYTHON_EXEC="${PYTHON_EXEC}"; then
    warn "cmake configuration failed; skipping pybullet_ompl build."
    popd >/dev/null
    popd >/dev/null
    popd >/dev/null
    return
  fi

  if ! micromamba run -n "${MPDLX_ENV_NAME}" make -j"$(nproc)" update_bindings; then
    warn "update_bindings target failed; continuing without regenerating bindings."
  fi

  if ! micromamba run -n "${MPDLX_ENV_NAME}" make -j"$(nproc)"; then
    warn "pybullet_ompl build failed; skipping installation."
    popd >/dev/null
    popd >/dev/null
    popd >/dev/null
    return
  fi
  popd >/dev/null
  popd >/dev/null

  if ! micromamba run -n "${MPDLX_ENV_NAME}" pip install -e .; then
    warn "pip install -e deps/pybullet_ompl failed."
  fi
  popd >/dev/null
}

install_precommit() {
  pushd "${MPDLX_REPO}" >/dev/null
  micromamba run -n "${MPDLX_ENV_NAME}" pre-commit install || true
  popd >/dev/null
}

ensure_safe_git
ensure_submodules
ensure_isaac
link_datasets
ensure_env
configure_shell_hook
build_pybullet_ompl
install_precommit

log "Devcontainer setup complete."
