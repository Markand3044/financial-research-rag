import os
import json

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq

from citation_validator import validate_citations
from citation_handler import get_source_pages, format_sources
from output_schema import RAGResponse


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
    "C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss",
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
# 7. CHUNK LOOKUP
# =========================================================

chunk_lookup = {
    document.metadata["chunk_id"]: document
    for document in documents
}


# =========================================================
# 8. CONTEXT EXPANSION
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
# 9. QUERY NORMALIZATION
# =========================================================

def normalize_query(query):

    normalized_query = query.strip()

    replacements = {
        "how many countries does infosys operate in":
            "Infosys operates in countries global presence",

        "how many employees did infosys have":
            "Infosys employees workforce total employees",

        "how many active clients did infosys have":
            "Infosys active clients number",

        "what percentage of infosys revenue came from north america":
            "Infosys revenue by geography North America percentage",

        "what is the purpose of infosys":
            "Infosys purpose amplify human potential",

        "what are the values represented by c-life":
            "Infosys C-LIFE values Client value Leadership Integrity Fairness Excellence",
    }

    query_lower = normalized_query.lower()

    if query_lower in replacements:
        return replacements[query_lower]

    return normalized_query


# =========================================================
# 9. RAG FUNCTION
# =========================================================

def run_rag(query):

    if not query or not query.strip():
        raise ValueError("Question cannot be empty")

    query = query.strip()

    retrieval_query = normalize_query(query)

    # -----------------------------------------------------
    # FAISS RETRIEVAL
    # -----------------------------------------------------

    faiss_results = vector_store.similarity_search(
        retrieval_query,
        k=20
    )

    # -----------------------------------------------------
    # BM25 RETRIEVAL
    # -----------------------------------------------------

    tokenized_query = retrieval_query.lower().split()

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

    # -----------------------------------------------------
    # COMBINE FAISS + BM25
    # -----------------------------------------------------

    candidate_documents = {}

    for document in faiss_results + bm25_results:

        key = (
            document.metadata["pdf_page"],
            document.metadata["chunk_id"]
        )

        candidate_documents[key] = document

    candidates = list(candidate_documents.values())

    # -----------------------------------------------------
    # BGE RERANKING
    # -----------------------------------------------------

    pairs = [
        [retrieval_query, document.page_content]
        for document in candidates
    ]

    scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------------------------------
    # SELECT TOP 5
    # -----------------------------------------------------

    top_results = reranked_results[:10]

    print("\n==============================")
    print("TOP RERANKED RESULTS")
    print("==============================")

    for rank, (document, score) in enumerate(
        top_results,
        start=1
    ):
        print(
            f"Rank {rank} | "
            f"Score {score:.4f} | "
            f"Page {document.metadata['pdf_page']} | "
            f"Chunk {document.metadata['chunk_id']}"
        )

    # -----------------------------------------------------
    # BUILD FINAL CONTEXT
    # -----------------------------------------------------

    # -----------------------------------------------------
    # BUILD FINAL CONTEXT
    # -----------------------------------------------------

    final_context = []
    seen_chunks = set()

    for document, score in top_results:

        # Temporary test:
        # Use only the BGE reranked chunks.
        # Do not expand with previous/next chunks.
        chunk_id = document.metadata["chunk_id"]

        if chunk_id not in seen_chunks:

            final_context.append(document)
            seen_chunks.add(chunk_id)
    # -----------------------------------------------------
    # BUILD LABELED CONTEXT
    # -----------------------------------------------------

    context_parts = []

    for document in final_context:

        chunk_id = document.metadata["chunk_id"]
        pdf_page = document.metadata["pdf_page"]

        context_parts.append(
            f"[CHUNK_{chunk_id}]\n"
            f"PDF Page: {pdf_page}\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    # -----------------------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------------------

    system_prompt = """
    You are a financial research assistant.

    Answer the user's question using ONLY the provided context.

    Return your response as valid JSON with exactly these two fields:

    {
        "answer": "your answer here",
        "citations": ["CHUNK_ID_1", "CHUNK_ID_2"]
    }

    Rules:
    1. The answer must be based only on the provided context.
    2. citations must contain only chunk IDs that actually appear in the context.
    3. Cite only chunks that directly support the answer.
    4. Do not invent chunk IDs.
    5. If the context does not contain enough information to answer the question, say so in the answer and return an empty citations list.
    6. Do not include PDF page numbers in the citations.
    7. Do not use Markdown outside the JSON object.
    """

    # -----------------------------------------------------
    # USER PROMPT
    # -----------------------------------------------------

    user_prompt = f"""
    Context from the annual report:

    {context}

    User question:

    {query}

    Answer the question using only the provided context.
    """

    # -----------------------------------------------------
    # CALL QWEN
    # -----------------------------------------------------

    try:

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",

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

            response_format={
                "type": "json_object"
            },

            temperature=0.1,

            max_tokens=1000
        )

    except Exception as error:

        print("\nLLM ERROR:")
        print(error)

        return {
            "question": query,
            "answer": "",
            "citation_ids": [],
            "validated_documents": [],
            "source_pages": [],
            "final_context": final_context,
            "error": str(error)
        }

    # -----------------------------------------------------
    # PARSE LLM RESPONSE
    # -----------------------------------------------------

    llm_output = response.choices[0].message.content

    try:

        result = json.loads(llm_output)

        # Validate the LLM output using Pydantic
        structured_response = RAGResponse.model_validate(result)

        answer = structured_response.answer
        citation_ids = structured_response.citations

    except (json.JSONDecodeError, ValueError) as error:

        print("\nSTRUCTURED OUTPUT ERROR:")
        print(error)

        answer = ""
        citation_ids = []

    # -----------------------------------------------------
    # VALIDATE CITATIONS
    # -----------------------------------------------------

    validated_documents = validate_citations(
        citation_ids,
        final_context
    )

    source_pages = get_source_pages(
        validated_documents
    )

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {
        "question": query,
        "answer": answer,
        "citation_ids": citation_ids,
        "validated_documents": validated_documents,
        "source_pages": source_pages,
        "final_context": final_context
    }


# =========================================================
# 10. MANUAL TEST MODE
# =========================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    ).strip()

    result = run_rag(query)

    print("\n")
    print("======================================")
    print("             RAG ANSWER")
    print("======================================")

    print(result["answer"])

    print("\nSources:")

    for page in result["source_pages"]:
        print(f"- PDF Page {page}")

    print("======================================")