from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Checkpoint:
    total_generated: int = 0
    p2p_count: int = 0
    merchant_count: int = 0
    seen_hashes_count: int = 0
    last_updated_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_generated": self.total_generated,
            "p2p_count": self.p2p_count,
            "merchant_count": self.merchant_count,
            "seen_hashes_count": self.seen_hashes_count,
            "last_updated_timestamp": self.last_updated_timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        return cls(
            total_generated=int(d.get("total_generated", 0)),
            p2p_count=int(d.get("p2p_count", 0)),
            merchant_count=int(d.get("merchant_count", 0)),
            seen_hashes_count=int(d.get("seen_hashes_count", 0)),
            last_updated_timestamp=str(d.get("last_updated_timestamp", "")),
        )


def default_checkpoint_path(output_dir: Optional[str] = None) -> Path:
    od = Path(output_dir or os.getenv("OUTPUT_DIR", "data"))
    od.mkdir(parents=True, exist_ok=True)
    return od / "checkpoint.json"


def load_checkpoint(path: Path) -> Checkpoint:
    if not path.exists():
        return Checkpoint()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Checkpoint.from_dict(data if isinstance(data, dict) else {})
    except Exception:
        return Checkpoint()


def save_checkpoint(path: Path, cp: Checkpoint) -> None:
    cp.last_updated_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cp.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

