from reg_pipeline_v2 import (
    vector_store,
    documents,
    bm25,
    reranker
)

query = "How many countries does Infosys operate in?"

# -----------------------------
# 1. FAISS candidates
# -----------------------------
faiss_results = vector_store.similarity_search(query, k=20)

# -----------------------------
# 2. BM25 candidates
# -----------------------------
tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(tokenized_query)

top_bm25_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:20]

bm25_results = [
    documents[index]
    for index in top_bm25_indices
]

# -----------------------------
# 3. Hybrid union
# -----------------------------
candidate_dict = {}

for document in faiss_results + bm25_results:

    key = (
        document.metadata["pdf_page"],
        document.metadata["chunk_id"]
    )

    candidate_dict[key] = document

candidates = list(candidate_dict.values())

print("\n" + "=" * 80)
print("TOTAL HYBRID CANDIDATES:", len(candidates))
print("=" * 80)

# -----------------------------
# 4. BGE reranking
# -----------------------------
pairs = [
    (query, document.page_content)
    for document in candidates
]

scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)

print("\n" + "=" * 80)
print("BGE RERANKING")
print("=" * 80)

for rank, (document, score) in enumerate(
    ranked_results,
    start=1
):

    page = document.metadata["pdf_page"]
    chunk_id = document.metadata["chunk_id"]

    marker = ""

    if chunk_id in [33, 34]:
        marker = "  <<< CORRECT COUNTRY EVIDENCE"

    print(
        f"Rank {rank:2d} | "
        f"Score {score:.4f} | "
        f"Page {page} | "
        f"Chunk {chunk_id}"
        f"{marker}"
    )