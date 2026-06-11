from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path
    output_jsonl: Path


def default_export_config(output_dir: Optional[str] = None) -> ExportConfig:
    od = Path(output_dir or os.getenv("OUTPUT_DIR", "data"))
    return ExportConfig(output_dir=od, output_jsonl=od / "output.jsonl")


class JSONLExporter:
    def __init__(self, config: ExportConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Open in append mode, line-buffered.
        self._fp = open(self.config.output_jsonl, "a", encoding="utf-8", buffering=1)

    def append(self, record: Dict[str, Any], fsync: bool = True) -> None:
        record = dict(record)
        record.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        line = json.dumps(record, ensure_ascii=False)
        self._fp.write(line + "\n")
        self._fp.flush()
        if fsync:
            os.fsync(self._fp.fileno())

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

