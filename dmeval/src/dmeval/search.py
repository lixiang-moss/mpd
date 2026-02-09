"""
Candidate generation for Stage I (tune) (SearchStrategy).

Design goals:
- Provide a minimal but extensible generator: grid / random (optuna reserved)
- The generated unit is a `Candidate`:
  - `candidate_id`: stable string ID (for on-disk organization and reproducibility)
  - `patch`: a YAML patch dict (merged into `base_cfg` to produce `cfg_inference.yaml`)
  - `params_flat`: flattened params dict (for logging/printing/debugging)

Notes:
- This module does not evaluate or rank candidates; it only generates them.
- In L1, the evaluation unit is the candidate (run across seeds, then aggregated and ranked).
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable

try:
    # After Hydra/OmegaConf composition, YAML lists are often `ListConfig` rather than native `list`.
    from omegaconf import ListConfig  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    ListConfig = ()  # type: ignore[assignment]

from .util import merge_patches, nested_dict_from_dotpath


@dataclass(frozen=True)
class Candidate:
    """One candidate configuration."""
    candidate_id: str
    patch: dict[str, Any]
    params_flat: dict[str, Any]


def _as_list(value: Any) -> list[Any]:
    """Wrap scalars into a single-item list to normalize grid search spaces."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if ListConfig and isinstance(value, ListConfig):
        return list(value)
    return [value]


def generate_candidates(*, base_patch: dict[str, Any], search_cfg: Any) -> list[Candidate]:
    """
    Generate candidates based on `search_cfg`.

    - `grid`: Cartesian product over discrete value lists
    - `random`: sampling from discrete choices or simple float/int ranges
    - `optuna`: reserved (not implemented in this minimal build)
    """
    strategy = str(getattr(search_cfg, "strategy", "grid")).lower()
    if strategy == "grid":
        space = getattr(search_cfg, "space", {})
        return list(_grid_candidates(base_patch=base_patch, space=space))
    if strategy == "random":
        space = getattr(search_cfg, "space", {})
        n_candidates = int(getattr(search_cfg, "n_candidates", 10))
        seed = getattr(search_cfg, "seed", None)
        return list(_random_candidates(base_patch=base_patch, space=space, n=n_candidates, seed=seed))
    if strategy == "optuna":
        raise NotImplementedError(
            "optuna strategy is not implemented in this minimal build; install dmeval[optuna] and extend search.py"
        )
    raise ValueError(f"Unknown search strategy: {strategy}")


def _grid_candidates(*, base_patch: dict[str, Any], space: Any) -> Iterable[Candidate]:
    """Grid search: Cartesian product over discrete lists per parameter."""
    if not isinstance(space, Mapping) or not space:
        yield Candidate(candidate_id="c0000", patch=base_patch, params_flat={})
        return

    keys = list(space.keys())
    values_lists = [_as_list(space[k]) for k in keys]

    for i, combo in enumerate(itertools.product(*values_lists)):
        params = dict(zip(keys, combo))
        patch_parts = [base_patch]
        for k, v in params.items():
            # Turn dotpath params like "a.b.c" into nested dict patches for YAML merge.
            patch_parts.append(nested_dict_from_dotpath(str(k), v))
        patch = merge_patches(*patch_parts)
        yield Candidate(candidate_id=f"c{i:04d}", patch=patch, params_flat=params)


def _random_candidates(*, base_patch: dict[str, Any], space: Any, n: int, seed: Any) -> Iterable[Candidate]:
    """Random search: sample `n` candidates from the given distributions/choices."""
    if seed is not None:
        random.seed(int(seed))

    if not isinstance(space, Mapping) or not space:
        for i in range(n):
            yield Candidate(candidate_id=f"c{i:04d}", patch=base_patch, params_flat={})
        return

    for i in range(n):
        params: dict[str, Any] = {}
        for key, spec in space.items():
            params[str(key)] = _sample_one(spec)
        patch_parts = [base_patch]
        for k, v in params.items():
            patch_parts.append(nested_dict_from_dotpath(k, v))
        patch = merge_patches(*patch_parts)
        yield Candidate(candidate_id=f"c{i:04d}", patch=patch, params_flat=params)


def _sample_one(spec: Any) -> Any:
    """
    Sample one value from a parameter spec.

    Supported spec forms:
    - list: choose uniformly from the list
    - dict:
      - {"choices": [...]} for discrete options
      - {"type": "int", "low": 1, "high": 10}
      - {"type": "float", "low": 1e-3, "high": 1e-1, "log": true}
    """
    if isinstance(spec, list):
        return random.choice(spec)
    if isinstance(spec, dict):
        kind = str(spec.get("type", "")).lower()
        if "choices" in spec:
            return random.choice(list(spec["choices"]))
        if kind in ("int", "integer"):
            low = int(spec["low"])
            high = int(spec["high"])
            return random.randint(low, high)
        if kind in ("float", "number"):
            low = float(spec["low"])
            high = float(spec["high"])
            if bool(spec.get("log", False)):
                if low <= 0 or high <= 0:
                    raise ValueError("log-uniform requires low/high > 0")
                return math.exp(random.uniform(math.log(low), math.log(high)))
            return random.uniform(low, high)
    return spec
