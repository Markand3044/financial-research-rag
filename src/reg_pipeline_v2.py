import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


# =========================================================
# 2. LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 3. LOAD FAISS VECTOR STORE
# =========================================================

print("Loading FAISS vector store...")

vector_store = FAISS.load_local(
    "data/vectorstore/infosys_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

documents = list(vector_store.docstore._dict.values())

print("Total documents:", len(documents))


# =========================================================
# 4. CREATE BM25 INDEX
# =========================================================

print("Creating BM25 index...")

tokenized_documents = [
    document.page_content.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)


# =========================================================
# 5. LOAD BGE RERANKER
# =========================================================

print("Loading BGE reranker...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# =========================================================
# 6. LOAD GROQ CLIENT
# =========================================================

client = Groq(
    api_key=groq_api_key
)


# =========================================================
# 7. USER QUESTION
# =========================================================

query = input("\nEnter your question: ").strip()

if not query:
    raise ValueError("Question cannot be empty")


# =========================================================
# 8. FAISS RETRIEVAL
# =========================================================

print("\nRunning FAISS retrieval...")

faiss_results = vector_store.similarity_search(
    query,
    k=20
)


# =========================================================
# 9. BM25 RETRIEVAL
# =========================================================

print("Running BM25 retrieval...")

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
# 10. COMBINE FAISS + BM25
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
# 11. BGE RERANKING
# =========================================================

print("Running BGE reranker...")

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
# 12. SELECT TOP 5
# =========================================================

top_results = reranked_results[:5]


print("\n==============================")
print("TOP RERANKED RESULTS")
print("==============================")

for rank, (document, score) in enumerate(top_results, start=1):

    print(
        f"Rank {rank} | "
        f"Score {score:.4f} | "
        f"Page {document.metadata['pdf_page']} | "
        f"Chunk {document.metadata['chunk_id']}"
    )


# =========================================================
# 13. CHUNK LOOKUP
# =========================================================

chunk_lookup = {
    document.metadata["chunk_id"]: document
    for document in documents
}


# =========================================================
# 14. CONTEXT EXPANSION
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
# 15. BUILD FINAL CONTEXT
# =========================================================

final_context = []
seen_chunks = set()

for document, score in top_results:

    expanded_chunks = expand_context(document)

    for chunk in expanded_chunks:

        chunk_id = chunk.metadata["chunk_id"]

        if chunk_id not in seen_chunks:

            final_context.append(chunk)
            seen_chunks.add(chunk_id)


print("\nFinal context chunks:", len(final_context))


# =========================================================
# 16. BUILD LLM CONTEXT
# =========================================================

context_parts = []

for document in final_context:

    page = document.metadata["pdf_page"]
    chunk_id = document.metadata["chunk_id"]

    context_parts.append(
        f"""
[PDF Page: {page} | Chunk: {chunk_id}]
{document.page_content}
"""
    )


context = "\n".join(context_parts)


# =========================================================
# 17. CREATE PROMPT
# =========================================================

system_prompt = """
You are a financial research assistant.

Answer the user's question using ONLY the information
provided in the context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts or numbers.
3. If the context does not contain enough information,
   say that the information is not available in the
   provided document.
4. Give a clear and concise answer.
5. When possible, mention the relevant PDF page number.
6. For numerical questions, preserve the numbers and
   units exactly as supported by the context.
7. Do not provide financial advice or tell the user
   whether they should buy or sell an investment.
"""


user_prompt = f"""
Context from the annual report:

{context}

User question:

{query}

Answer the question using only the provided context.
"""


# =========================================================
# 18. CALL QWEN 3.6 27B
# =========================================================

print("\nGenerating answer with Qwen 3.6 27B...")

response = client.chat.completions.create(
    model="qwen/qwen3.6-27b",

    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],

    reasoning_format="hidden",

    temperature=0.1,

    max_tokens=1000
)


# =========================================================
# 19. GET ANSWER
# =========================================================

answer = response.choices[0].message.content


# =========================================================
# 20. COLLECT SOURCE PAGES
# =========================================================

source_pages = sorted(
    {
        document.metadata["pdf_page"]
        for document in final_context
    }
)


# =========================================================
# 21. DISPLAY FINAL ANSWER
# =========================================================

print("\n")
print("======================================")
print("             RAG ANSWER")
print("======================================")

print(answer)

print("\nSources:")

for page in source_pages:
    print(f"- PDF Page {page}")

print("======================================")