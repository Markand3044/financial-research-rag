import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

with open("C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_chunks_langchain.json", 'r', encoding= 'utf - 8') as file:
    chunks = json.load(file)

print("chunks loaded :-", len(chunks))

Doc = []

for chunk in chunks:

    document = Document(
        page_content = chunk['text'],
        metadata = chunk['metadata']
    )

    Doc.append(document)

print("Documnets created :", len(Doc))

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(Doc, embedding)

print("FAISS vector store created")

vector_store.save_local("C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss")

print("FAISS vector store saved.")