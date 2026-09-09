from sentence_transformers import CrossEncoder

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

embedding = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss",embedding, allow_dangerous_deserialization=True
)

documents = list(vector_store.docstore._dict.values())

print("Total Document :", len(documents))

tokenized_documents = [
    document.page_content.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

question = "How many active clients did Infosys have?"

# -----------------------------------
# 6. FAISS retrieval
# -----------------------------------

faiss_results = vector_store.similarity_search(
    question,
    k=20
)


# -----------------------------------
# 7. BM25 retrieval
# -----------------------------------

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
# 8. Combine candidates
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


print(
    "\nCandidate pool:",
    len(combined_documents)
)


# -----------------------------------
# 9. Load BGE reranker
# -----------------------------------

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# -----------------------------------
# 10. Create question-document pairs
# -----------------------------------

pairs = [
    [question, document.page_content]
    for document in combined_documents
]


# -----------------------------------
# 11. Calculate relevance scores
# -----------------------------------

scores = reranker.predict(pairs)


# -----------------------------------
# 12. Sort by score
# -----------------------------------

ranked_results = sorted(
    zip(combined_documents, scores),
    key=lambda x: x[1],
    reverse=True
)


# -----------------------------------
# 13. Display reranked results
# -----------------------------------

print("\n==============================")
print("       BGE RERANKED")
print("==============================\n")


for rank, (document, score) in enumerate(
    ranked_results[:10],
    start=1
):

    print(
        f"Rank {rank} | "
        f"Score {score:.4f} | "
        f"Page {document.metadata['pdf_page']}"
    )

    print(
        document.page_content[:300]
    )

    print("-" * 80)