from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


_WS_RE = re.compile(r"\s+")
_HAS_PAYMENT_RAIL_RE = re.compile(r"\b(UPI|IMPS|NEFT|RTGS|POS)\b", re.IGNORECASE)
_HAS_DATEISH_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4}|\d{4}\d{2}\d{2})\b")
_HAS_UPI_HANDLE_RE = re.compile(r"@[a-z0-9._-]{2,}", re.IGNORECASE)
_GENERIC_BAD_RE = re.compile(
    r"\b(lorem ipsum|here is|as an ai|example narration|transaction narration)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationConfig:
    min_len: int = 20
    max_len: int = 240


def normalize_for_hash(s: str) -> str:
    s2 = s.strip()
    s2 = _WS_RE.sub(" ", s2)
    return s2


def validate_narration(narration: str, txn_type: str, cfg: ValidationConfig | None = None) -> Tuple[bool, str]:
    cfg = cfg or ValidationConfig()
    txn_type = txn_type.upper().strip()
    if txn_type not in {"P2P", "MERCHANT"}:
        return False, "invalid_txn_type"

    if not narration or not narration.strip():
        return False, "empty"

    s = narration.strip().strip('"').strip("'").strip()
    if "\n" in s or "\r" in s:
        return False, "multiline"

    if len(s) < cfg.min_len:
        return False, "too_short"
    if len(s) > cfg.max_len:
        return False, "too_long"

    if _GENERIC_BAD_RE.search(s):
        return False, "generic_llm_text"

    # Must look like a bank narration.
    if not _HAS_PAYMENT_RAIL_RE.search(s):
        return False, "missing_payment_rail"

    # Prefer typical bank-style separators.
    if "/" not in s and "-" not in s:
        return False, "missing_separators"

    # Encourage realistic texture: ref/date/handles.
    score = 0
    if _HAS_UPI_HANDLE_RE.search(s):
        score += 1
    if _HAS_DATEISH_RE.search(s):
        score += 1
    if re.search(r"\b(REF|TXN|TRF|PMT|PAY|PAID)\b", s, re.IGNORECASE):
        score += 1
    if score == 0:
        return False, "not_banklike_enough"

    if txn_type == "MERCHANT":
        # Merchant narrations often carry a brand-ish token; at least require absence of obvious person-only pattern.
        if _HAS_UPI_HANDLE_RE.search(s) and re.search(r"\b(rahul|priya|ankit|sneha|apoorv|mahendr)\b", s, re.I):
            # Still allow, but nudge away from pure P2P by rejecting the clearest cases.
            return False, "looks_like_p2p"

    return True, "ok"

