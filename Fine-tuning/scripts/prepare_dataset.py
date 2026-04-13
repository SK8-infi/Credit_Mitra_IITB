import argparse
import json
import random
from pathlib import Path


SYSTEM_INSTRUCTION = (
    "You are an information extraction model. Extract only the payee name from the transaction narration. "
    "Return only the payee text, with no extra words."
)


def build_prompt(narration: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Transaction narration:\n{narration}\n\n"
        f"Payee:"
    )


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input labels.jsonl path")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    raw = load_jsonl(input_path)
    examples = []
    for row in raw:
        narration = str(row.get("narration", "")).strip()
        payee = str(row.get("payee", "")).strip()
        if not narration or not payee:
            continue

        examples.append(
            {
                "id": row.get("id"),
                "type": row.get("type"),
                "prompt": build_prompt(narration),
                "response": payee,
            }
        )

    random.seed(args.seed)
    random.shuffle(examples)

    val_count = int(len(examples) * args.val_ratio)
    val_rows = examples[:val_count]
    train_rows = examples[val_count:]

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)

    print(f"Total: {len(examples)}")
    print(f"Train: {len(train_rows)}")
    print(f"Val: {len(val_rows)}")
    print(f"Wrote: {out_dir / 'train.jsonl'}")
    print(f"Wrote: {out_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
