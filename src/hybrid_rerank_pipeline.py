import json

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# =========================================================
# 1. LOAD EMBEDDING MODEL
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 2. LOAD FAISS VECTOR STORE
# =========================================================

vector_store = FAISS.load_local(
    "data/vectorstore/infosys_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

documents = list(vector_store.docstore._dict.values())

print("Total documents:", len(documents))


# =========================================================
# 3. CREATE BM25 INDEX
# =========================================================

tokenized_documents = [
    document.page_content.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)


# =========================================================
# 4. LOAD BGE RERANKER
# =========================================================

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# =========================================================
# 5. QUERY
# =========================================================

query = "How many active clients did Infosys have?"


# =========================================================
# 6. FAISS RETRIEVAL
# =========================================================

faiss_results = vector_store.similarity_search(
    query,
    k=20
)


# =========================================================
# 7. BM25 RETRIEVAL
# =========================================================

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(tokenized_query)

top_bm25_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:20]

bm25_results = [
    documents[i]
    for i in top_bm25_indices
]


# =========================================================
# 8. COMBINE FAISS + BM25
# =========================================================

candidate_documents = {}

for document in faiss_results + bm25_results:

    key = (
        document.metadata["pdf_page"],
        document.metadata["chunk_id"]
    )

    candidate_documents[key] = document


candidates = list(candidate_documents.values())

print("Candidate pool:", len(candidates))


# =========================================================
# 9. BGE RERANKING
# =========================================================

pairs = [
    [query, document.page_content]
    for document in candidates
]

scores = reranker.predict(pairs)


reranked_results = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)


# =========================================================
# 10. SELECT TOP 5
# =========================================================

top_results = reranked_results[:5]


print("\n==============================")
print("BGE RERANKED RESULTS")
print("==============================")

for rank, (document, score) in enumerate(top_results, start=1):

    print(
        f"\nRank {rank} | "
        f"Score {score:.4f} | "
        f"Page {document.metadata['pdf_page']} | "
        f"Chunk {document.metadata['chunk_id']}"
    )

    print(document.page_content[:500])


# =========================================================
# 11. CREATE CHUNK LOOKUP
# =========================================================

chunk_lookup = {
    document.metadata["chunk_id"]: document
    for document in documents
}


# =========================================================
# 12. CONTEXT EXPANSION
# =========================================================

def expand_context(document):

    chunk_id = document.metadata["chunk_id"]
    page = document.metadata["pdf_page"]

    expanded = []

    # Previous chunk
    previous_document = chunk_lookup.get(chunk_id - 1)

    if previous_document:
        if previous_document.metadata["pdf_page"] == page:
            expanded.append(previous_document)

    # Current chunk
    expanded.append(document)

    # Next chunk
    next_document = chunk_lookup.get(chunk_id + 1)

    if next_document:
        if next_document.metadata["pdf_page"] == page:
            expanded.append(next_document)

    return expanded


# =========================================================
# 13. BUILD FINAL CONTEXT
# =========================================================

final_context = []

for document, score in top_results:

    expanded_chunks = expand_context(document)

    for chunk in expanded_chunks:

        if chunk not in final_context:
            final_context.append(chunk)


# =========================================================
# 14. DISPLAY FINAL CONTEXT
# =========================================================

print("\n==============================")
print("FINAL EXPANDED CONTEXT")
print("==============================")

for i, document in enumerate(final_context, start=1):

    print(
        f"\nContext {i} | "
        f"Page {document.metadata['pdf_page']} | "
        f"Chunk {document.metadata['chunk_id']}"
    )

    print(document.page_content[:500])


print("\n==============================")
print("TOTAL FINAL CONTEXT CHUNKS:", len(final_context))
print("==============================")