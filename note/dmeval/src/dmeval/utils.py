from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Optional, Tuple


def to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)

    # Optional torch support.
    try:
        import torch  # type: ignore

        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
    except Exception:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_nan(value: float) -> bool:
    return math.isnan(value)


def format_template(text: str, variables: Dict[str, str]) -> str:
    try:
        return text.format(**variables)
    except KeyError:
        return text


def as_plain_mapping(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return dict(obj)
    to_dict = getattr(obj, "toDict", None)
    if callable(to_dict):
        return to_dict()
    if hasattr(obj, "__dict__"):
        return dict(getattr(obj, "__dict__"))
    return None


def flatten_mapping(mapping: Mapping[str, Any], *, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def _walk(node: Mapping[str, Any], parent_key: str) -> None:
        for k, v in node.items():
            full_key = f"{parent_key}.{k}" if parent_key else str(k)
            if isinstance(v, Mapping):
                _walk(v, full_key)
                continue
            out[full_key] = v

    _walk(mapping, prefix)
    return out


def nanmean(values: Iterable[float]) -> float:
    xs = [v for v in values if not is_nan(v)]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def nanstd(values: Iterable[float]) -> float:
    xs = [v for v in values if not is_nan(v)]
    if len(xs) < 2:
        return float("nan")
    mean = sum(xs) / len(xs)
    var = sum((v - mean) ** 2 for v in xs) / (len(xs) - 1)
    return float(math.sqrt(var))


_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.=-]+")


def safe_token(text: str) -> str:
    text = str(text).strip()
    text = _SAFE_TOKEN_RE.sub("_", text)
    return text.strip("_")


def format_value_for_tag(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        s = f"{value:.6g}"
        return s.replace("-", "m").replace(".", "p")
    return safe_token(str(value))


def set_by_dotted_key(mapping: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = str(dotted_key).split(".")
    if not parts:
        raise ValueError("Empty key.")
    cur: Dict[str, Any] = mapping
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def nested_from_dotted_map(values: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in values.items():
        set_by_dotted_key(out, k, v)
    return out
