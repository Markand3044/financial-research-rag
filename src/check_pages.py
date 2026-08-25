import json

input_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_pages.json"

output_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_cleaned_pages.json"

# Load original extracted pages
with open(
    input_path,
    "r",
    encoding="utf-8"
) as file:
    original_pages = json.load(file)


# Load cleaned pages
with open(
    output_path,
    "r",
    encoding="utf-8"
) as file:
    cleaned_pages = json.load(file)


# Get page numbers from both files
original_page_numbers = {
    page["pdf_page"]
    for page in original_pages
}

cleaned_page_numbers = {
    page["pdf_page"]
    for page in cleaned_pages
}


# Find pages that disappeared
missing_pages = original_page_numbers - cleaned_page_numbers


print("Missing PDF pages:", missing_pages)