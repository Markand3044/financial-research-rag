import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

input_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_cleaned_pages.json"
output_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_chunks_langchain.json"

with open(input_path, 'r', encoding= 'utf-8') as file:
    pages = json.load(file)

print("Number of pages :-", len(pages))

text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)

chunks = []

chunk_id = 0

for page in pages :
    page_chunk = text_spliter.split_text(page['text'])

    for chunk_text in page_chunk:   

        chunks.append({
            'text': chunk_text,
            'metadata' : {
                "company":'Infosys',
                "document" : "Infosys Integrated Annual Report 2025-2026",
                "financial_year" : "2025-2026",
                "pdf_page" : page['pdf_page'],
                "pdf_page": page["pdf_page"],
                "chunk_id": chunk_id
            }
        })

        chunk_id += 1 

with open(output_path, 'w', encoding='utf -8') as file:
    json.dump(chunks, file, ensure_ascii= False, indent=4)

print("Total chunks :-", len(chunks))
print("saved chunks to :-", output_path)

print('\n--- FIRST CHUNK ---\n')
print(chunks[0]['text'])

print("\n----- METADATA -----\n")
print(chunks[0]["metadata"])
