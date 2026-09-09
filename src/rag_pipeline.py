import os 

from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

groq_api_key = os.getenv("GRQ_API_KEY")

client = Groq(
    api_key=groq_api_key
)

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vectore_store = FAISS.load_local("C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss",
                                 embedding,
                                 allow_dangerous_deserialization=True)

question = "what was infosys total revenue in fiscal 2026 ?"

results = vectore_store.similarity_search(question, k=3)

context_parts = []

for document in results:
    context_parts.append(
        f"""
Source :{document.metadata}

Content : {document.page_content}
"""
    )

context = "\n\n".join(context_parts)

prompt = f"""
You are a financial research assistant.

Answer the user's question ONLY using the information
provided in the context below.

Rules:
1. Do not invent financial information.
2. If the context does not contain enough information,
   clearly say that the information is not available
   in the provided documents.
3. Give a concise and factual answer.
4. Mention the relevant source page when possible.

User Question:
{question}

Context:
{context}
"""

response = client.chat.completions.create(
    model="qwen/qwen3.6-27b",
    messages=[
        {
            "role": "system",
            "content": "You are a careful financial research assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    reasoning_format="hidden"
)


# --------------------------------------------------
# 10. Display final answer
# --------------------------------------------------

answer = response.choices[0].message.content

print("\n==============================")
print("       RAG ANSWER")
print("==============================\n")

print(answer)


# --------------------------------------------------
# 11. Display sources
# --------------------------------------------------

print("\n==============================")
print("       SOURCES")
print("==============================\n")

for i, document in enumerate(results, start=1):

    print(f"Source {i}:")
    print(document.metadata)
    print()