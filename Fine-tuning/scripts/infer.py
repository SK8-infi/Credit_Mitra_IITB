import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(text: str) -> str:
    return (
        "You are an information extraction model. Extract only the payee name from the transaction narration. "
        "Return only the payee text, with no extra words.\n\n"
        f"Transaction narration:\n{text}\n\n"
        "Payee:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()

    prompt = build_prompt(args.text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = full_text[len(prompt) :].strip()
    print(pred)


if __name__ == "__main__":
    main()
