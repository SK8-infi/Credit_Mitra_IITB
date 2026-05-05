# from extract_transactions import extract_transactions_using_llm
from docling_extraction_from_tabular_transaction import extract_transaction_from_pdf

text = extract_transaction_from_pdf("transactions.pdf")

# output = extract_transactions_using_llm(text)
# print(output)
print("Docling extracted text length:", len(text))

with open("docling_output.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Saved Docling extraction to docling_output.txt")