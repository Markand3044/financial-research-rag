from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local("C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss", embedding,allow_dangerous_deserialization=True)

documents = list(
    vector_store.docstore._dict.values()
    )

print("Total documents :", len(documents))

tokenized_documents = [
    document.page_content.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

question = "How many active clients did Infosys have?"

faiss_results = vector_store.similarity_search(question, k = 20)

query_tokens = question.lower().split()

bm25_scores = bm25.get_scores(query_tokens)

top_bm25_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:20]


bm25_results = [
    documents[i]
    for i in top_bm25_indices
]

# -----------------------------------
# 8. Display FAISS results
# -----------------------------------

print("\n==============================")
print("       FAISS TOP 10")
print("==============================")

for rank, document in enumerate(
    faiss_results[:10],
    start=1
):

    print(
        f"Rank {rank} | "
        f"Page {document.metadata['pdf_page']}"
    )


# -----------------------------------
# 9. Display BM25 results
# -----------------------------------

print("\n==============================")
print("       BM25 TOP 10")
print("==============================")

for rank, document in enumerate(
    bm25_results[:10],
    start=1
):

    print(
        f"Rank {rank} | "
        f"Page {document.metadata['pdf_page']}"
    )


# -----------------------------------
# 10. Combine candidates
# -----------------------------------

combined_documents = []

seen = set()

for document in faiss_results + bm25_results:

    key = (
        document.metadata.get("pdf_page"),
        document.page_content
    )

    if key not in seen:

        combined_documents.append(document)
        seen.add(key)


print("\n==============================")
print("    HYBRID CANDIDATE POOL")
print("==============================")

print(
    "Total unique candidates:",
    len(combined_documents)
)


for rank, document in enumerate(
    combined_documents,
    start=1
):

    print(
        f"Candidate {rank} | "
        f"Page {document.metadata['pdf_page']}"
    )   