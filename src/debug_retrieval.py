import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local("C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss", embedding, allow_dangerous_deserialization=True)

questions = [
    "How many active clients did Infosys have ?",
    "What are the values represented by C-LIFE ?"
]

for question in questions:

    print("\n" + "="*100)
    print("Question:", question)
    print("="*100)

    results = vector_store.similarity_search(question, k = 10)

    for i, document in enumerate(results, start=1):

        print(f"\n RESULT {i}")
        print("-" * 80)

        print("PDF PAGE :", document.metadata['pdf_page'])

        print("\nTEXT:")
        print(document.page_content)