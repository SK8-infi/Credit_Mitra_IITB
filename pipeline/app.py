"""
Credit Mitra – PDF → Structured Transactions Pipeline
=====================================================
Upload a bank-statement PDF.  Docling extracts tables,
payee-lora (via Ollama) predicts payee names, and you download enriched JSON.

Each inference is a fresh, stateless Ollama /api/generate call (no context field).
"""

import os
import re
import tempfile
from pathlib import Path

import httpx
import uvicorn
from docling.document_converter import DocumentConverter
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# ── Ollama config ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "payee-lora:latest")

SYSTEM_INSTRUCTION = (
    "You are an information extraction model. Extract only the payee name "
    "from the transaction narration. Return only the payee text, with no extra words."
)


def build_prompt(narration: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Transaction narration:\n{narration}\n\n"
        f"Payee:"
    )


def predict_payee(narration: str, client: httpx.Client) -> str:
    """Stateless payee extraction via Ollama (no conversation context)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_prompt(narration),
        "stream": False,
        "raw": True,
        "options": {"temperature": 0, "num_predict": 32},
    }
    # Never send "context" — each transaction is an independent request.
    resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def normalize_narration(text: str) -> str:
    """Collapse multi-line narrations into one line; segments join with no gap."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [line.strip() for line in text.split("\n") if line.strip()]
    if parts:
        return "".join(parts)
    return re.sub(r"\s+", " ", text).strip()


# ── Docling extraction ───────────────────────────────────────────────────
def extract_transactions(pdf_path: str) -> list[dict]:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    rows: list[dict] = []
    for table in result.document.tables:
        df = table.export_to_dataframe()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.fillna("")

        for _, row in df.iterrows():
            rec = {
                "date": str(row.get("date", "")).strip(),
                "particulars": normalize_narration(str(row.get("particulars", ""))),
                "deposits": str(row.get("deposits", "")).strip(),
                "withdrawals": str(row.get("withdrawals", "")).strip(),
                "balance": str(row.get("balance", "")).strip(),
            }
            if not any(rec.values()):
                continue
            rows.append(rec)
    return rows


# ── FastAPI ──────────────────────────────────────────────────────────────
app = FastAPI(title="Credit Mitra Pipeline")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/process-pdf")
async def process_pdf(pdf: UploadFile = File(...)):
    suffix = Path(pdf.filename or "upload.pdf").suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await pdf.read()
    tmp.write(content)
    tmp.close()

    try:
        transactions = extract_transactions(tmp.name)

        with httpx.Client(timeout=120.0) as client:
            for txn in transactions:
                narration = txn.get("particulars", "")
                if narration and narration not in ("Opening Balance", "Closing Balance"):
                    txn["payee"] = predict_payee(narration, client)
                else:
                    txn["payee"] = ""

        return JSONResponse(content={"status": "success", "transactions": transactions})
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    print(f"Ollama: {OLLAMA_BASE_URL}  model={OLLAMA_MODEL}")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
