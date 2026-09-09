import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

with open(
    "C:/Users/Admin/Desktop/FinancialResearchRAG/data/evaluation/questions.json",
    'r',
    encoding = 'utf-8'
) as file:
    questions = json.load(file)



embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = FAISS.load_local("C:/Users/Admin/Desktop/FinancialResearchRAG/data/vectorstore/infosys_faiss.json",
                                embedding,
                                allow_dangerous_deserialization=True)


print("Total questions :", len(questions))

print("\n==============================")
print(" RETRIEVAL EVALUATION")
print("==============================\n")

for index, item in enumerate(questions, start = 1):

    question = item["question"]
    expected_page = item["expected_page"]

    results = vector_store.similarity_search(question, k = 5)

    retrived_pages = [
        document.metadata['pdf_page']
        for document in results
    ]

    found = expected_page in retrived_pages

    print(f"Question {index}: {question}")
    print(f"Expected page : {expected_page}")
    print(f"Retrival_page : {retrived_pages}")
    print(f"Result : {'PASS' if found else 'Fail'} ")
    print("-"*70)