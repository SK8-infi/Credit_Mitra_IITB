from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from checkpoint import Checkpoint, default_checkpoint_path, load_checkpoint, save_checkpoint
from exporter import default_export_config
from llm_client import GeminiNarrationClient


log = logging.getLogger("labeler")


@dataclass
class LabelCheckpoint:
    total_labeled: int = 0
    last_updated_timestamp: str = ""

    def to_dict(self) -> Dict:
        return {
            "total_labeled": self.total_labeled,
            "last_updated_timestamp": self.last_updated_timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LabelCheckpoint":
        return cls(
            total_labeled=int(d.get("total_labeled", 0)),
            last_updated_timestamp=str(d.get("last_updated_timestamp", "")),
        )


def default_label_checkpoint_path(output_dir: Optional[str] = None) -> Path:
    base = Path(output_dir or os.getenv("OUTPUT_DIR", "data"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "label_checkpoint.json"


def load_label_checkpoint(path: Path) -> LabelCheckpoint:
    if not path.exists():
        return LabelCheckpoint()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LabelCheckpoint.from_dict(data if isinstance(data, dict) else {})
    except Exception:
        return LabelCheckpoint()


def save_label_checkpoint(path: Path, cp: LabelCheckpoint) -> None:
    import time as _time

    cp.last_updated_timestamp = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cp.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def rebuild_labeled_ids(labels_path: Path) -> Set[int]:
    """
    Rebuild the set of already-labeled record IDs from labels.jsonl.
    """
    ids: Set[int] = set()
    if not labels_path.exists():
        return ids
    with labels_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            try:
                rid = int(obj.get("id"))
            except Exception:
                continue
            ids.add(rid)
    return ids


def load_narrations(output_jsonl: Path) -> List[Dict]:
    """
    Load all narrations from output.jsonl, keeping id + narration + type.
    """
    rows: List[Dict] = []
    if not output_jsonl.exists():
        return rows
    with output_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "narration" not in obj or "id" not in obj:
                continue
            rows.append(
                {
                    "id": int(obj["id"]),
                    "narration": str(obj["narration"]),
                    "type": str(obj.get("type", "")).upper(),
                }
            )
    rows.sort(key=lambda r: r["id"])
    return rows


PAYEE_EXTRACTION_INSTRUCTIONS = """You will be given a single Indian bank transaction narration text.

Your task is to extract ONLY the payee name.

Definitions:
- If money is SENT (debit), the person or entity receiving the money is the PAYEE.
- If money is RECEIVED (credit), the person or entity who sent the money is the PAYEE.

Rules:
- Extract the payee name exactly as it appears in the narration.
- Do NOT hallucinate or invent names.
- Ignore amounts, timestamps, banks, account numbers, reference IDs, and UPI handles themselves (like @okaxis).
- For UPI, the name may appear near 'to', 'from', 'via', 'upi', or the VPA, but return only the human-readable name or merchant label, not the raw handle.
- For POS/Card, the merchant name is the payee.
- For IMPS/NEFT/RTGS, the counterparty name is the payee.

Output format:
- Return ONLY the payee name as plain text.
- No quotes, no explanation, no JSON.
- If you truly cannot identify any payee name, return an empty string.
"""


def build_label_prompt(narration: str) -> str:
    return PAYEE_EXTRACTION_INSTRUCTIONS + "\n\nTransaction narration:\n" + narration.strip()


@dataclass(frozen=True)
class LabelConfig:
    progress_log_every: int = 50


class PayeeLabeler:
    def __init__(
        self,
        output_dir: Optional[str] = None,
        output_jsonl: Optional[str] = None,
        labels_filename: str = "labels.jsonl",
    ):
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "data")
        export_cfg = default_export_config(self.output_dir)
        self.output_path = Path(output_jsonl or export_cfg.output_jsonl)
        self.labels_path = Path(self.output_dir) / labels_filename

        # Reuse the same Gemini client; you can override model via GEMINI_MODEL.
        self.llm = GeminiNarrationClient()
        self.cfg = LabelConfig()

        # Resume state.
        self.labeled_ids: Set[int] = rebuild_labeled_ids(self.labels_path)
        self.label_ckpt_path = default_label_checkpoint_path(self.output_dir)
        self.label_ckpt = load_label_checkpoint(self.label_ckpt_path)

        # Also load generation checkpoint so we know how many items exist.
        self.gen_ckpt_path = default_checkpoint_path(self.output_dir)
        self.gen_ckpt: Checkpoint = load_checkpoint(self.gen_ckpt_path)

        self.rows: List[Dict] = load_narrations(self.output_path)
        self._state_lock = asyncio.Lock()

    async def _label_one(self, row: Dict) -> Dict:
        prompt = build_label_prompt(row["narration"])
        try:
            raw = await self.llm.generate_one(prompt)
            payee = raw.strip().splitlines()[0].strip()
        except Exception as e:  # noqa: BLE001
            # If the API key is blocked, fail fast so the user can rotate the key.
            msg = str(e).lower()
            if "api key was reported as leaked" in msg or "leaked" in msg and "api key" in msg:
                raise RuntimeError(
                    "Gemini API key is blocked: reported as leaked. Please replace GEMINI_API_KEY in .env with a new key."
                ) from e
            if "permission_denied" in msg or "403" in msg:
                raise RuntimeError(
                    "Gemini API call failed with 403 PERMISSION_DENIED. Check GEMINI_API_KEY permissions/quotas."
                ) from e
            # For transient/validation failures, label as empty and continue training data pipeline.
            payee = ""

        rec = {
            "id": row["id"],
            "type": row["type"],
            "narration": row["narration"],
            "payee": payee,
            "model": self.llm.config.model,
        }

        # Append atomically.
        os.makedirs(self.output_dir, exist_ok=True)
        with self.labels_path.open("a", encoding="utf-8", buffering=1) as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        return rec

    async def run(self) -> None:
        if not self.rows:
            log.warning("No narrations found in %s; nothing to label.", self.output_path)
            return

        to_process = [r for r in self.rows if r["id"] not in self.labeled_ids]
        if not to_process:
            log.info("All %s narrations already labeled.", len(self.rows))
            return

        log.info(
            "Starting labeling: total_narrations=%s already_labeled=%s remaining=%s",
            len(self.rows),
            len(self.labeled_ids),
            len(to_process),
        )

        async def worker(worker_id: int) -> None:
            # Simple work-stealing over an index: avoids deadlocks on fatal errors.
            while True:
                async with self._state_lock:
                    idx = getattr(self, "_next_label_idx", 0)
                    if idx >= len(to_process):
                        return
                    setattr(self, "_next_label_idx", idx + 1)
                row = to_process[idx]
                rec = await self._label_one(row)
                async with self._state_lock:
                    self.labeled_ids.add(rec["id"])
                    self.label_ckpt.total_labeled = len(self.labeled_ids)
                    save_label_checkpoint(self.label_ckpt_path, self.label_ckpt)
                    if self.label_ckpt.total_labeled % self.cfg.progress_log_every == 0:
                        log.info("labeled=%s / %s", self.label_ckpt.total_labeled, len(self.rows))

        # Use the same max concurrency as the generator's LLM client.
        workers = [
            asyncio.create_task(worker(i))
            for i in range(max(1, self.llm.config.max_concurrency))
        ]
        await asyncio.gather(*workers)
        log.info("Labeling complete: labeled=%s / %s", self.label_ckpt.total_labeled, len(self.rows))


async def run_labeler(output_dir: Optional[str] = None, output_jsonl: Optional[str] = None) -> None:
    labeler = PayeeLabeler(output_dir=output_dir, output_jsonl=output_jsonl)
    await labeler.run()

