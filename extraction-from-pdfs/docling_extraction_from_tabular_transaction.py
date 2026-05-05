from docling.document_converter import DocumentConverter
import json
import pandas as pd

def extract_transaction_from_pdf(pdf_path):
    print("Starting Docling PDF conversion...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    
    all_transactions = []
    
    for table in result.document.tables:
        df = table.export_to_dataframe()
        
        # Normalize column names to lowercase to make parsing robust
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Convert DataFrame to list of dicts.
        # fillna("") ensures that pd.NaT/pd.NaN are replaced with empty strings
        df_filled = df.fillna("")
        row_dicts = df_filled.to_dict(orient="records")
        
        for row in row_dicts:
            transaction = {
                "date": str(row.get("date", "")).strip(),
                "particulars": str(row.get("particulars", "")).strip(),
                "deposits": str(row.get("deposits", "")).strip(),
                "withdrawals": str(row.get("withdrawals", "")).strip(),
                "balance": str(row.get("balance", "")).strip()
            }
            
            # Remove rows that are entirely empty across all 5 fields
            if not any(transaction.values()):
                continue
            
            all_transactions.append(transaction)
            
    json_output = json.dumps(all_transactions, indent=4)
    print(f"Extracted {len(all_transactions)} table rows to JSON.")
    return json_output