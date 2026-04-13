import argparse
import json
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`.,;:!?()\[\]{}]", "", text)
    return text


def jaccard_token_similarity(a: str, b: str) -> float:
    a_set = set(normalize_text(a).split())
    b_set = set(normalize_text(b).split())
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def build_prompt(narration: str) -> str:
    return (
        "You are an information extraction model. Extract only the payee name from the transaction narration. "
        "Return only the payee text, with no extra words.\n\n"
        f"Transaction narration:\n{narration}\n\n"
        "Payee:"
    )


@dataclass
class EvalRow:
    id: str
    narration: str
    gold: str
    pred: str
    exact_match: int
    normalized_exact_match: int
    char_similarity: float
    token_jaccard: float


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model(base_model: str, lora_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, narration: str, max_new_tokens: int = 32) -> str:
    prompt = build_prompt(narration)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return full[len(prompt) :].strip()


def summarize(rows):
    total = len(rows)
    if total == 0:
        return {
            "samples": 0,
            "exact_match": 0.0,
            "normalized_exact_match": 0.0,
            "avg_char_similarity": 0.0,
            "avg_token_jaccard": 0.0,
        }
    return {
        "samples": total,
        "exact_match": sum(r.exact_match for r in rows) / total,
        "normalized_exact_match": sum(r.normalized_exact_match for r in rows) / total,
        "avg_char_similarity": mean(r.char_similarity for r in rows),
        "avg_token_jaccard": mean(r.token_jaccard for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--val-file", type=str, default="data/val.jsonl")
    parser.add_argument("--out-dir", type=str, default="outputs/eval")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all")
    args = parser.parse_args()

    val_rows = load_jsonl(Path(args.val_file))
    if args.max_samples > 0:
        val_rows = val_rows[: args.max_samples]

    model, tokenizer = load_model(args.base_model, args.lora_path)

    eval_rows = []
    for row in tqdm(val_rows, desc="Evaluating"):
        narration = row.get("prompt", "")
        # Backward compatibility: when val file has prompt/response from prepare_dataset.py
        if "Transaction narration:\n" in narration and "\n\nPayee:" in narration:
            narration = narration.split("Transaction narration:\n", 1)[1].split("\n\nPayee:", 1)[0]

        gold = str(row.get("response", "")).strip()
        pred = predict(model, tokenizer, narration, max_new_tokens=args.max_new_tokens)

        em = int(pred == gold)
        nem = int(normalize_text(pred) == normalize_text(gold))

        eval_rows.append(
            EvalRow(
                id=str(row.get("id", "")),
                narration=narration,
                gold=gold,
                pred=pred,
                exact_match=em,
                normalized_exact_match=nem,
                char_similarity=char_similarity(pred, gold),
                token_jaccard=jaccard_token_similarity(pred, gold),
            )
        )

    metrics = summarize(eval_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    preds_path = out_dir / "predictions.jsonl"
    errors_path = out_dir / "errors_top20.jsonl"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with preds_path.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    errors = [r for r in eval_rows if r.normalized_exact_match == 0]
    errors.sort(key=lambda x: x.char_similarity)
    with errors_path.open("w", encoding="utf-8") as f:
        for r in errors[:20]:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {preds_path}")
    print(f"Worst errors: {errors_path}")


if __name__ == "__main__":
    main()
