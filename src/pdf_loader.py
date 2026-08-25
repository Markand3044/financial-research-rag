from pypdf import PdfReader
import json

path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/raw/infosys-ar-26.pdf"

read = PdfReader(path)

print("Number of pages :-", len(read.pages))

pages1 = []

for page_number, page in enumerate(read.pages, start=1):

    text = page.extract_text()

    pages1.append({
        'pdf_page': page_number,
        'text': text
    })

print("Pages extracted:", len(pages1))

output_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_pages.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(pages1, file, ensure_ascii= False, indent= 4)
print("saved extracted pages to:", output_path)