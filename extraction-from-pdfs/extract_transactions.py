# from groq import Groq
# from .models import TransactionList
# from pydantic import ValidationError
import json5
import re
import os

from dotenv import load_dotenv
load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
def extract_json(text: str):
    fenced = re.search(r"```json(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    array = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
    if array:
        return array.group(0).strip()

    obj = re.search(r"{.*}", text, re.DOTALL)
    if obj:
        return obj.group(0).strip()

    print("⚠️ No JSON found, returning empty list")
    return []


def repair_json_using_llm(text: str):
    # client = Groq(
    #     api_key=GROQ_API_KEY)

    prompt = f"""
Fix the following broken JSON and return ONLY valid strict JSON.
Do not add or remove any fields. Do not explain anything.
The data here is tabular just the orientation is distorted , their are fixed fields treat it as table.
Return ONLY the corrected JSON array.

{text}
"""

    # response = client.chat.completions.create(
    #     model="openai/gpt-oss-120b",
    #     temperature=0,
    #     messages=[{"role": "user", "content": prompt}],
    # )

    # fixed_output = response.choices[0].message.content
    # return extract_json(fixed_output)
    print("Groq disabled. Returning empty list.")
    return []


def extract_transactions_using_llm(text: str):
    # client = Groq(
    #     api_key=GROQ_API_KEY)

    prompt = f"""
Extract all bank transactions from the following text.
Return ONLY valid JSON.

Fields:
- date
- amount
- type
- balance
- reference_number
- category implies wheter it is UPI /NEFT /RTGS etc , you can find it from the transaction
- transaction

If any field is missing, set it to null.
Return ONLY a JSON array.

Text:
{text}
"""

    # response = client.chat.completions.create(
    #     model="openai/gpt-oss-120b",
    #     temperature=0,
    #     messages=[{"role": "user", "content": prompt}],
    # )

    # raw_output = response.choices[0].message.content
    raw_output = "[]"

    print("TYPE:", type(raw_output))
    print("LENGTH:", len(raw_output) if raw_output else 0)
    print("OUTPUT:", raw_output[:500])
    json_str = extract_json(raw_output)
    print(json_str)
    print("\n====== RAW LLM OUTPUT ======\n")
    print(raw_output)
    print("\n===========================\n")

    repaired = repair_json_using_llm(json_str)
    print(repaired)
    return repaired
