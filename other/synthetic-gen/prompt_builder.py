from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


P2P_FORMAT_HINTS = [
    "UPI/DR/<upi>/<desc>",
    "UPI/CR/<upi>/<desc>",
    "IMPS/<name>/<ref>",
    "NEFT/<name>/<bank>/<ref>",
]

MERCHANT_FORMAT_HINTS = [
    "UPI/DR/<merchant>/<order_or_ref>",
    "POS/<merchant>/<txn_or_ref>",
    "UPI/DR/<merchant>/<short_desc>/<ref>",
]

COMMON_UPI_HANDLES = ["okaxis", "oksbi", "okhdfcbank", "ybl", "ibl", "paytm", "upi"]

INDIAN_NAMES = [
    "Rahul Sharma",
    "Priya Verma",
    "Ankit Gupta",
    "Sneha Nair",
    "Apoorva Singh",
    "Mahendra Patil",
    "Raseel Khan",
    "Yash Nigam",
    "Daulat Rao",
    "Kiran Mehta",
    "Saurabh Jain",
    "Neha Kulkarni",
    "Vivek Iyer",
    "Pooja Batra",
    "Rohit Tiwari",
    "Amit Sinha",
    "Nikhil Arora",
    "Isha Menon",
    "Ritika Chawla",
    "Arjun Reddy",
    "Tanvi Desai",
    "Harshit Bansal",
    "Nandini Roy",
    "Gaurav Malhotra",
    "Shivani Das",
    "Raghav Bhatia",
    "Lakshmi Pillai",
    "Manoj Yadav",
    "Aarav Kapoor",
    "Meera Joshi",
]

MERCHANTS = [
    "Swiggy",
    "Zomato",
    "Ola",
    "Uber",
    "Amazon",
    "Flipkart",
    "Meesho",
    "JioMart",
    "DMart",
    "IRCTC",
    "Myntra",
    "BookMyShow",
    "BigBasket",
    "Blinkit",
    "Reliance Fresh",
    "Starbucks",
    "Cafe Coffee Day",
    "Dominos",
    "Pizza Hut",
    "Apollo Pharmacy",
    "MedPlus",
    "Nykaa",
    "Ajio",
    "Croma",
    "Vijay Sales",
    "Paytm Mall",
    "MakeMyTrip",
    "Goibibo",
    "Air India",
    "IndiGo",
    "RedBus",
    "Uber Eats",
    "Westside",
    "Pantaloons",
]

NOISE_TOKENS = [
    "TRF",
    "TXN",
    "REF",
    "PMT",
    "PAY",
    "PAID VIA",
    "PAYMENT",
    "UPI TRAN",
    "EXPRESS",
    "ONLY RS",
]


@dataclass(frozen=True)
class PromptConfig:
    few_shot_min: int = 3
    few_shot_max: int = 6


class PromptBuilder:
    def __init__(self, real_examples: Dict[str, Sequence[str]], config: PromptConfig | None = None):
        self.real_examples_a = [e.strip() for e in real_examples.get("format_a", []) if e and e.strip()]
        self.real_examples_b = [e.strip() for e in real_examples.get("format_b", []) if e and e.strip()]
        self.real_examples = self.real_examples_a + self.real_examples_b
        if len(self.real_examples_a) < 3 or len(self.real_examples_b) < 3:
            raise ValueError("Need examples from both CSV formats (at least 3 each)")
        if len(self.real_examples_a) + len(self.real_examples_b) < 8:
            raise ValueError("Need at least 5 real examples for style reference")
        self.config = config or PromptConfig()

    def build(self, txn_type: str) -> str:
        txn_type = txn_type.upper().strip()
        if txn_type not in {"P2P", "MERCHANT"}:
            raise ValueError("txn_type must be P2P or MERCHANT")

        k = random.randint(self.config.few_shot_min, self.config.few_shot_max)
        k_a = max(1, k // 2)
        k_b = max(1, k - k_a)
        examples = random.sample(self.real_examples_a, k=min(k_a, len(self.real_examples_a))) + random.sample(
            self.real_examples_b, k=min(k_b, len(self.real_examples_b))
        )
        random.shuffle(examples)

        fmt_hint = random.choice(P2P_FORMAT_HINTS if txn_type == "P2P" else MERCHANT_FORMAT_HINTS)
        noise = ", ".join(random.sample(NOISE_TOKENS, k=random.randint(2, 5)))
        upi_hint = f"{random.choice(INDIAN_NAMES).lower()}{random.randint(10,9999)}@{random.choice(COMMON_UPI_HANDLES)}"
        merchant_hint = random.choice(MERCHANTS)

        extra_constraints = []
        extra_constraints.append(f"- Suggested format hint: {fmt_hint}")
        extra_constraints.append("- Output must be a single line (no quotes, no bullets).")
        extra_constraints.append("- Keep it realistic for Indian banking SMS/statement narration.")
        extra_constraints.append("- Add small noise: mixed casing, truncation, extra slashes/spaces, abbreviations.")
        extra_constraints.append(f"- Sprinkle abbreviations like: {noise}")
        if txn_type == "P2P":
            extra_constraints.append(f"- Use realistic Indian person name/UPI handle (e.g. {upi_hint}).")
        else:
            extra_constraints.append(f"- Use realistic Indian merchant brand (e.g. {merchant_hint}).")

        few_shot = "\n".join(f"- {e}" for e in examples)

        return (
            "You are generating and labeling one Indian bank transaction narration sample.\n\n"
            "Based on the examples below, generate ONE highly realistic transaction narration and the correct payee.\n\n"
            f"Constraints:\n"
            f"- Type: {txn_type}\n"
            "- Follow Indian banking formats (UPI/IMPS/NEFT/RTGS/POS)\n"
            "- Use realistic names, UPI IDs, merchants\n"
            "- Avoid repetition and templates\n"
            "- Output MUST be valid JSON with exactly two keys: narration, payee\n"
            "- payee must be the exact beneficiary/merchant/person receiving money in the narration\n"
            "- payee must be clean text name only: no @handle, no bank suffix, no trailing digits\n"
            "- narration must contain enough cues so payee is unambiguous\n"
            + "\n".join(extra_constraints)
            + "\n\n"
            "Return ONLY JSON. No markdown. No explanation.\n"
            'Example output: {"narration":"UPI/DR/...","payee":"BigBasket"}\n\n'
            "Examples:\n"
            f"{few_shot}\n"
        )


def load_examples_from_csv_rows(rows: List[dict]) -> Dict[str, List[str]]:
    # Expected columns: Category, Narration.
    # The user-provided file has two formats in order; keep both and use equally.
    narrations = [(r.get("Narration") or "").strip() for r in rows]
    narrations = [n for n in narrations if n]
    split_idx = max(1, len(narrations) // 2)
    format_a = narrations[:split_idx]
    format_b = narrations[split_idx:]
    if not format_b:
        format_b = format_a[:]
    return {"format_a": format_a, "format_b": format_b}

