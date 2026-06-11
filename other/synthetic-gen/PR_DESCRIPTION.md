# 🏦 feat: Add Synthetic Bank Transaction Narration Generator

## Summary

Added a production-grade synthetic data generation pipeline that produces highly realistic Indian bank transaction narrations. The system analyses patterns from real bank statement narrations and replicates them at scale using **Gemma 27B** (hosted on Google AI Studio via the Gemini API).

---

## Motivation

Training and testing downstream models (fraud detection, categorisation, analytics) requires large volumes of realistic transaction data. Real bank narrations are sensitive and scarce. This pipeline bridges that gap by learning the style, structure, and noise characteristics of actual Indian bank narrations and generating synthetic equivalents that are statistically indistinguishable from the originals.

---

## What Was Done

1. **Pattern Analysis** — Studied 25 real Indian bank statement narrations to identify formatting conventions, field ordering, abbreviation patterns, UPI ID structures, reference ID formats, and common noise (truncation, inconsistent casing, whitespace irregularities).

2. **Pipeline Construction** — Built a modular, fault-tolerant generation pipeline with the following components:

   | Module | Responsibility |
   |---|---|
   | `llm_client.py` | Gemini API wrapper with exponential backoff, retry logic, and rate-limit throttling |
   | `prompt_builder.py` | Per-transaction prompt construction with randomised few-shot examples and transaction-type enforcement |
   | `validator.py` | Output validation — rejects empty, malformed, or generic responses |
   | `checkpoint.py` | Crash-safe resume system with per-write checkpoint persistence |
   | `exporter.py` | Append-only JSONL writer for zero data-loss guarantees |
   | `generator.py` | Async orchestration loop with concurrency control and deduplication |
   | `main.py` | CLI entrypoint with configurable parameters |

3. **Generation Strategy** — Each API call generates **exactly one** narration (no batching). The system enforces a **90% P2P / 10% merchant** distribution via dynamic prompt adjustment and maintains a hash-based deduplication set to ensure uniqueness across the entire corpus.

4. **Indian Financial Context** — Narrations follow real-world Indian banking formats:
   - **UPI**: `UPI/DR/<merchant>/<bank>/<upi_id>/<desc>//<ref>/<timestamp>`
   - **IMPS / NEFT / RTGS**: with realistic names, bank codes, and reference IDs
   - **Merchants**: Swiggy, Flipkart, IRCTC, Jio, Groww, Goibibo, etc.
   - **Noise**: abbreviations (`TRF`, `TXN`, `PMT`), truncated descriptions, mixed casing

---

## Key Design Decisions

- **One LLM call per narration** — strict constraint; no batching to maintain maximum diversity and control.
- **Append-only persistence** — every accepted narration is written to `data/output.jsonl` and checkpoint updated *immediately*, so no data is lost on crash or interruption.
- **Resume-safe** — on restart the pipeline rebuilds its dedup set from the existing output file and continues where it left off.
- **Async with concurrency cap** — parallel API calls (configurable, default 8) with RPM throttling to respect Gemini rate limits.

---

## Files Added

```
├── main.py               # CLI entrypoint
├── generator.py           # Async generation loop & orchestration
├── llm_client.py          # Gemini API client (retry + rate limiting)
├── prompt_builder.py      # Dynamic prompt construction
├── validator.py           # Narration validation rules
├── checkpoint.py          # Resume / checkpoint persistence
├── exporter.py            # Append-safe JSONL writer
├── transactions.csv       # 25 seed narrations (style reference)
├── requirements.txt       # Python dependencies
├── README.md              # Setup & usage docs
└── data/
    ├── output.jsonl       # Generated narrations (append-only)
    └── checkpoint.json    # Generation state for resume
```

---

## How to Run

```bash
# Install
pip install -r requirements.txt

# Set API key
$env:GEMINI_API_KEY="<your-key>"

# Generate
python main.py --input transactions.csv --total 10000
```

---

## Testing & Validation

- Pipeline has been run and validated with **1,385 narrations** generated successfully so far.
- Resume functionality verified — process can be stopped and restarted without data loss or duplicate generation.
- Output distribution tracking confirms adherence to the 90/10 P2P/merchant target split.
