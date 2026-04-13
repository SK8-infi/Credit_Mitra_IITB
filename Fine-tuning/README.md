# Payee Name Extraction with LoRA Fine-Tuning

This folder contains everything needed to fine-tune a small language model (SLM) to extract `payee` from transaction `narration`.

## 1) Goal

Given a narration string, predict only the payee name.

Example:
- Input: `IMPS/DR/.../PMT-Groceries-BigBasket/...`
- Output: `BigBasket`

## 2) Recommended base model (SLM)

Start with:
- `Qwen/Qwen2.5-1.5B-Instruct` (good quality/speed balance)

Alternative:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (lighter but lower accuracy)

## 3) Setup

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4) Prepare dataset

Convert your source file to train/val instruction datasets:

```bash
python scripts/prepare_dataset.py --input "..\other\synthetic-gen\data\labels.jsonl" --out-dir data --val-ratio 0.1 --seed 42
```

Generated files:
- `data/train.jsonl`
- `data/val.jsonl`

Each row has:
- `prompt`: strict extraction instruction + narration
- `response`: payee label

## 5) Train with LoRA

Basic run:

```bash
python scripts/train_lora.py --train-file data/train.jsonl --val-file data/val.jsonl --output-dir outputs/payee-lora
```

Notes:
- Uses 4-bit quantized loading + LoRA adapters (memory efficient).
- Good default targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

## 6) Evaluate quickly

```bash
python scripts/infer.py --base-model Qwen/Qwen2.5-1.5B-Instruct --lora-path outputs/payee-lora --text "UPI/P2P/.../rahul2601@ibl/..."
```

Full validation evaluation (exact + normalized metrics):

```bash
python scripts/evaluate.py --base-model Qwen/Qwen2.5-1.5B-Instruct --lora-path outputs/payee-lora --val-file data/val.jsonl --out-dir outputs/eval
```

Outputs:
- `outputs/eval/metrics.json`
- `outputs/eval/predictions.jsonl`
- `outputs/eval/errors_top20.jsonl`

## 7) Iteration plan

1. Run baseline training with defaults.
2. Check exact-match accuracy on `data/val.jsonl`.
3. Inspect wrong predictions:
   - wrong casing
   - extra text in answer
   - confusion between sender and payee
4. Improve data instruction and add harder synthetic patterns.
5. Re-train with:
   - lower LR (for stability)
   - 2-4 epochs
   - gradient accumulation if VRAM is limited
6. Freeze best adapter and version it.

## 8) Production behavior constraints

Prompt policy should enforce:
- Return only payee text.
- No explanation.
- Empty string if truly unknown.

If you want, next I can add:
- an evaluation script for exact match + fuzzy score,
- a tiny API (`FastAPI`) to serve inference,
- a script to merge LoRA into base model for deployment.
