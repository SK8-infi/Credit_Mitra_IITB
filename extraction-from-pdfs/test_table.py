from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert("transactions.pdf")
print("Number of tables:", len(result.document.tables))
for i, table in enumerate(result.document.tables):
    df = table.export_to_dataframe()
    print(f"Table {i} cols:", df.columns.tolist())
    print(df.head(2))
