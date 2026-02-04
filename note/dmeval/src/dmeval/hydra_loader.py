from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def load_config(config_path: str, *, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load a YAML config via Hydra compose (scheme A: Hydra in DMEval only).

    This supports:
    - plain YAML files
    - Hydra defaults/group composition
    - runtime overrides (Hydra override syntax)

    Notes:
    - We use Hydra's compose API (not @hydra.main), so we do not change cwd and we do not create Hydra outputs dirs.
    """
    try:
        from hydra import compose, initialize_config_dir  # type: ignore
        from omegaconf import OmegaConf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Hydra is required. Install dmeval with hydra-core>=1.3.") from exc

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    config_dir = str(path.parent)
    config_name = path.stem

    overrides = list(overrides or [])

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)

    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Expected a mapping config at the root.")
    return data

