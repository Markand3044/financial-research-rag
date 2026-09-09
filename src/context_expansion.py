import json 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local("C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss", embeddings, allow_dangerous_deserialization= True)

documents = list(vector_store.docstore._dict.values())
print("Total documents:", len(documents))

chunk_lookup ={
    doc.metadata["chunk_id"]: doc
    for doc in documents

}

# -----------------------------
# Context expansion function
# -----------------------------

def expand_context(selected_doc):
    
    chunk_id = selected_doc.metadata["chunk_id"]
    page = selected_doc.metadata["pdf_page"]

    expanded_chunks = []

    # Previous chunk
    previous_doc = chunk_lookup.get(chunk_id - 1)

    if previous_doc:
        if previous_doc.metadata["pdf_page"] == page:
            expanded_chunks.append(previous_doc)

    # Selected chunk
    expanded_chunks.append(selected_doc)

    # Next chunk
    next_doc = chunk_lookup.get(chunk_id + 1)

    if next_doc:
        if next_doc.metadata["pdf_page"] == page:
            expanded_chunks.append(next_doc)

    return expanded_chunks


# -----------------------------
# Test
# -----------------------------

query = "How many active clients did Infosys have?"

results = vector_store.similarity_search(query, k=5)

print("\n--- ORIGINAL RETRIEVED CHUNKS ---")

for doc in results:

    print(
        "Page:",
        doc.metadata["pdf_page"],
        "| Chunk:",
        doc.metadata["chunk_id"]
    )


print("\n--- EXPANDED CONTEXT ---")

for doc in results[:2]:

    expanded = expand_context(doc)

    print("\nSelected chunk:", doc.metadata["chunk_id"])

    for context_doc in expanded:

        print(
            "Page:",
            context_doc.metadata["pdf_page"],
            "| Chunk:",
            context_doc.metadata["chunk_id"]
        )