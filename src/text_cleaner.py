import json
import re

input_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_pages.json"

output_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_cleaned_pages.json"

with open(input_path, 'r', encoding= 'utf-8') as file:
    pages = json.load(file)

print("pages loaded :", len(pages))

cleand_pages = []

for page in pages:
    text = page['text']

    if text is None:
        text = ""

    # Remove unwanted soft-hyphen characters
    text = text.replace("\u00ad", "")

    # Fix words broken by a line break
    # Example: reverse-
    #          engineered
    # becomes: reverse-engineered

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1-\2",
        text
    )

    # Replace line breaks with spaces
    text = re.sub(r"\s*\n\s*", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary spaces at beginning/end
    text = text.strip()

    # Keep the page only if text exists
    if text:

        cleand_pages.append({
            "pdf_page": page['pdf_page'],
            "text": text
        })

with open(output_path, 'w', encoding = 'utf-8') as file:

    json.dump(cleand_pages, file, ensure_ascii= False, indent=4)

print("Cleaned pages:", len(cleand_pages))

print("Saved cleaned data to:", output_path)