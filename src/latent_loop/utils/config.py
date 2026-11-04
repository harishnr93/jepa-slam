"""YAML configuration loading with inheritance and command-line overrides.

A config file may contain ``extends: <other.yaml>`` (relative to its own directory).
The parent is loaded first and the child is deep-merged on top of it. Overrides use
dotted keys, e.g. ``dataset.sampling.stride=5``, and values are parsed as YAML scalars.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict where ``override`` is recursively merged into ``base``."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_with_extends(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular 'extends' chain detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    parent_name = data.pop("extends", None)
    if parent_name is None:
        return data
    parent = _load_with_extends(path.parent / parent_name, seen)
    return deep_merge(parent, data)


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` for ``dotted_key = "a.b.c"``, creating levels."""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def get_by_dotted_key(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for key in dotted_key.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def parse_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    """Parse ``key.path=value`` strings into a nested dict."""
    result: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(result, key.strip(), yaml.safe_load(raw_value))
    return result


def load_config(
    path: str | Path | None = None, overrides: Iterable[str] | None = None
) -> dict[str, Any]:
    """Load a YAML config (default: ``configs/default.yaml``) and apply overrides."""
    config_path = Path(path) if path is not None else CONFIG_DIR / "default.yaml"
    if not config_path.is_absolute() and not config_path.exists():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = _load_with_extends(config_path, set())
    if overrides:
        config = deep_merge(config, parse_overrides(overrides))
    config["_config_path"] = str(config_path.resolve())
    return config


def resolve_path(path: str | Path | None) -> Path | None:
    """Resolve a config path relative to the repository root."""
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path
