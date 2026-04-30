from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path | None, base: Path | None = None) -> Path | None:
    if path is None:
        return None
    if str(path).startswith("/path/to"):
        raise ValueError(
            f"Config still contains placeholder path {path!r}. "
            "Set it to a real writable path, e.g. under "
            "/ceph/behrens/ellie/language-decoding-expts/libribrain_gtr_decode."
        )
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base or project_root()) / p


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path, Path.cwd())
    if config_path is None:
        raise ValueError("Config path is required")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def config_hash(config: dict[str, Any]) -> str:
    clean = {k: v for k, v in config.items() if not str(k).startswith("_")}
    payload = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def output_dir(config: dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config["project"]["output_dir"]))


def cache_dir(config: dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config["project"].get("cache_dir"), output_dir(config)))


def ensure_dir(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Path is required")
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def sidecar_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_suffix(p.suffix + ".info.json")


def write_cache_info(path: str | Path, config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    info = {"config_hash": config.get("_config_hash"), "path": str(path)}
    if extra:
        info.update(extra)
    save_json(info, sidecar_path(path))


def cache_is_current(path: str | Path, config: dict[str, Any]) -> bool:
    p = Path(path)
    info_path = sidecar_path(p)
    if not p.exists() or not info_path.exists():
        return False
    try:
        return load_json(info_path).get("config_hash") == config.get("_config_hash")
    except Exception:
        return False
