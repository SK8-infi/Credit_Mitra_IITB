from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from tqdm import tqdm

from checkpoint import Checkpoint, default_checkpoint_path, load_checkpoint, save_checkpoint
from exporter import JSONLExporter, default_export_config
from llm_client import GeminiNarrationClient
from prompt_builder import PromptBuilder, load_examples_from_csv_rows
from validator import normalize_for_hash, validate_narration


log = logging.getLogger("generator")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_real_examples(input_csv: str) -> Dict[str, list[str]]:
    p = Path(input_csv)
    with p.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    return load_examples_from_csv_rows(rows)


def parse_model_json(raw: str) -> Dict[str, Any]:
    txt = raw.strip()
    # Try direct JSON first.
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: extract first {...} block.
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(txt[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def clean_payee(raw_payee: str) -> str:
    s = (raw_payee or "").strip()
    if not s:
        return ""
    # Remove common UPI handle-like substrings.
    s = re.sub(r"[a-z0-9._-]+@[a-z0-9._-]+", "", s, flags=re.IGNORECASE)
    # Remove remaining handle markers if model returns partials.
    s = re.sub(r"@\S+", "", s)
    # Drop digits glued to end of names (e.g. "manoj yadav5305").
    s = re.sub(r"(?<=[A-Za-z])\d{2,}$", "", s)
    # Keep readable name/merchant text.
    s = re.sub(r"\s+", " ", s).strip(" -_/,:;.")
    return s


def rebuild_state_from_output(output_jsonl: Path) -> Tuple[Set[str], Checkpoint]:
    """
    Rebuild dedupe hashes and counts from existing output file.
    This is the authoritative resume source.
    """
    seen: Set[str] = set()
    cp = Checkpoint()
    if not output_jsonl.exists():
        return seen, cp

    with output_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            narration = str(obj.get("narration", "")).strip()
            txn_type = str(obj.get("type", "")).upper().strip()
            if not narration or txn_type not in {"P2P", "MERCHANT"}:
                continue
            h = _sha1(normalize_for_hash(narration).lower())
            seen.add(h)
            cp.total_generated += 1
            if txn_type == "P2P":
                cp.p2p_count += 1
            else:
                cp.merchant_count += 1
    cp.seen_hashes_count = len(seen)
    return seen, cp


@dataclass(frozen=True)
class GenerationConfig:
    target_total: int = 10_000
    merchant_ratio: float = 0.10
    max_attempts_per_item: int = 30
    progress_log_every: int = 50


class NarrationGenerator:
    def __init__(
        self,
        input_csv: str,
        output_dir: Optional[str] = None,
        target_total: Optional[int] = None,
    ):
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "data")
        self.export_cfg = default_export_config(self.output_dir)
        self.ckpt_path = default_checkpoint_path(self.output_dir)

        self.real_examples = load_real_examples(input_csv)
        self.prompt_builder = PromptBuilder(self.real_examples)
        self.llm = GeminiNarrationClient()

        self.cfg = GenerationConfig(
            target_total=int(target_total or int(os.getenv("TARGET_TOTAL", "10000"))),
        )

        self.exporter = JSONLExporter(self.export_cfg)
        self.labels_jsonl = self.export_cfg.output_dir / "labels.jsonl"

        # Resume sources:
        self.seen_hashes, rebuilt = rebuild_state_from_output(self.export_cfg.output_jsonl)
        on_disk = load_checkpoint(self.ckpt_path)
        # Output file is authoritative; checkpoint is advisory.
        self.checkpoint = rebuilt
        if on_disk.total_generated != rebuilt.total_generated:
            log.warning(
                "Checkpoint mismatch; using output.jsonl counts. checkpoint=%s rebuilt=%s",
                on_disk.to_dict(),
                rebuilt.to_dict(),
            )

        self._state_lock = asyncio.Lock()
        self._start_time = time.monotonic()
        self._pbar: Optional[tqdm] = None

    def close(self) -> None:
        self.exporter.close()

    def _should_force_merchant(self) -> bool:
        # Enforce merchant minimum progressively so we converge to the final ratio.
        next_total = self.checkpoint.total_generated + 1
        required_merchant = int(self.cfg.merchant_ratio * next_total)
        return self.checkpoint.merchant_count < required_merchant

    def _pick_type(self) -> str:
        if self._should_force_merchant():
            return "MERCHANT"
        return "MERCHANT" if (os.urandom(1)[0] / 255.0) < self.cfg.merchant_ratio else "P2P"

    async def _generate_one_accepted(self, item_id: int) -> Dict:
        for attempt in range(1, self.cfg.max_attempts_per_item + 1):
            async with self._state_lock:
                txn_type = self._pick_type()
                prompt = self.prompt_builder.build(txn_type)

            t0 = time.monotonic()
            raw = await self.llm.generate_one(prompt)
            latency = time.monotonic() - t0

            obj = parse_model_json(raw)
            narration = str(obj.get("narration", "")).strip()
            payee = clean_payee(str(obj.get("payee", "")).strip())

            ok, reason = validate_narration(narration, txn_type)
            if not ok:
                log.info("reject item=%s attempt=%s type=%s reason=%s text=%r", item_id, attempt, txn_type, reason, narration[:120])
                continue
            if not payee or len(payee) < 2 or len(payee) > 120:
                log.info("reject item=%s attempt=%s type=%s reason=bad_payee text=%r", item_id, attempt, txn_type, payee[:120])
                continue

            h = _sha1(normalize_for_hash(narration).lower())
            async with self._state_lock:
                if h in self.seen_hashes:
                    log.info("duplicate item=%s attempt=%s type=%s", item_id, attempt, txn_type)
                    continue

                # Accept: persist immediately, then update in-memory & checkpoint.
                rec = {
                    "id": item_id,
                    "type": txn_type,
                    "narration": narration,
                    "payee": payee,
                    "hash": h,
                    "attempt": attempt,
                    "latency_s": round(latency, 3),
                    "model": self.llm.config.model,
                }
                self.exporter.append(rec, fsync=True)
                with self.labels_jsonl.open("a", encoding="utf-8") as lf:
                    lf.write(json.dumps({"id": item_id, "type": txn_type, "narration": narration, "payee": payee}, ensure_ascii=False) + "\n")

                self.seen_hashes.add(h)
                self.checkpoint.total_generated += 1
                if txn_type == "P2P":
                    self.checkpoint.p2p_count += 1
                else:
                    self.checkpoint.merchant_count += 1
                self.checkpoint.seen_hashes_count = len(self.seen_hashes)
                save_checkpoint(self.ckpt_path, self.checkpoint)
                if self._pbar is not None:
                    self._pbar.update(1)
                    self._pbar.set_postfix(
                        {
                            "p2p": self.checkpoint.p2p_count,
                            "merchant": self.checkpoint.merchant_count,
                        },
                        refresh=False,
                    )
                return rec

        raise RuntimeError(f"Failed to generate valid unique narration after {self.cfg.max_attempts_per_item} attempts")

    async def run(self) -> None:
        if self.checkpoint.total_generated >= self.cfg.target_total:
            log.info("Already complete: %s/%s", self.checkpoint.total_generated, self.cfg.target_total)
            return

        log.info(
            "Resume state: total=%s p2p=%s merchant=%s output=%s",
            self.checkpoint.total_generated,
            self.checkpoint.p2p_count,
            self.checkpoint.merchant_count,
            str(self.export_cfg.output_jsonl),
        )

        first_error: Dict[str, BaseException] = {}
        error_event = asyncio.Event()

        async def worker(worker_id: int, q: asyncio.Queue[int]) -> None:
            while True:
                item_id = await q.get()
                try:
                    if error_event.is_set():
                        return
                    async with self._state_lock:
                        if self.checkpoint.total_generated >= self.cfg.target_total:
                            return
                    await self._generate_one_accepted(item_id)

                    async with self._state_lock:
                        if self.checkpoint.total_generated % self.cfg.progress_log_every == 0:
                            elapsed = max(1e-6, time.monotonic() - self._start_time)
                            speed = self.checkpoint.total_generated / elapsed
                            pct = 100.0 * self.checkpoint.total_generated / self.cfg.target_total
                            log.info(
                                "progress=%.2f%% total=%s p2p=%s merchant=%s speed=%.2f txn/s",
                                pct,
                                self.checkpoint.total_generated,
                                self.checkpoint.p2p_count,
                                self.checkpoint.merchant_count,
                                speed,
                            )
                finally:
                    q.task_done()
                await asyncio.sleep(0)
                # Continue until producer is done or target reached.
        async def guarded_worker(worker_id: int, q: asyncio.Queue[int]) -> None:
            try:
                await worker(worker_id, q)
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # noqa: BLE001
                if not error_event.is_set():
                    first_error["err"] = e
                    error_event.set()
                raise

        start_id = self.checkpoint.total_generated
        remaining = self.cfg.target_total - self.checkpoint.total_generated

        # Queue a bounded number of pending IDs to avoid memory blowups.
        q: asyncio.Queue[int] = asyncio.Queue(maxsize=500)
        self._pbar = tqdm(
            total=self.cfg.target_total,
            initial=self.checkpoint.total_generated,
            desc="Generating narrations",
            unit="txn",
            dynamic_ncols=True,
        )
        workers = [
            asyncio.create_task(guarded_worker(i, q))
            for i in range(max(1, self.llm.config.max_concurrency))
        ]

        try:
            for item_id in range(start_id, start_id + remaining):
                async with self._state_lock:
                    if self.checkpoint.total_generated >= self.cfg.target_total:
                        break
                if error_event.is_set():
                    break
                await q.put(item_id)

            await q.join()
        finally:
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            for t in workers:
                t.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            if "err" in first_error:
                raise first_error["err"]


async def run_generator(input_csv: str, total: int, output_dir: Optional[str] = None) -> None:
    gen = NarrationGenerator(input_csv=input_csv, output_dir=output_dir, target_total=total)
    try:
        await gen.run()
    finally:
        gen.close()

