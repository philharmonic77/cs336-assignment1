import json
from copy import deepcopy
from typing import Any
from pathlib import Path
from typing import Any

def _parse(s: str) -> Any:
    sl = s.lower()
    if sl in {"none", "null"}:
        return None
    if sl in {"true", "false"}:
        return sl == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # 支持 (0.9,0.999) 这种写法
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if not inner:
            return tuple()
        return tuple(_parse(p.strip()) for p in inner.split(",") if p.strip())
    return s  # 默认当字符串

def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")

        k, v = item.split("=", 1)
        keys = k.strip().split(".")
        cur: Any = out

        for kk in keys[:-1]:
            if not isinstance(cur, dict):
                raise ValueError(f"Override path '{k}' crosses a non-dict node at '{kk}'")
            if kk not in cur:
                cur[kk] = {}
            cur = cur[kk]

        if not isinstance(cur, dict):
            raise ValueError(f"Override path '{k}' targets a non-dict parent")
        cur[keys[-1]] = _parse(v.strip())

    return out

def save_run_config(out_dir: Path, cfg: dict[str, Any], overrides: list[str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 保存 config
    cfg_path = out_dir / "config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    # 2) 保存 overrides（可选，但很有用）
    if overrides is not None:
        ov_path = out_dir / "overrides.txt"
        ov_path.write_text("\n".join(overrides) + ("\n" if overrides else ""), encoding="utf-8")