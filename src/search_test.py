from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local("C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss",
                                embedding,
                                allow_dangerous_deserialization=True)

question = "what was infosys total revenue in fiscal 2026?"

result = vector_store.similarity_search(question, k=3)

print("\n --- SEARCH RESULT ---- \n")

for i, document in enumerate(result, start=1):

    print(f"RESULT {i}")

    print("\nText:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

    print("\n" + "-" * 80)
    