import json

from reg_pipeline_v2 import run_rag
from citation_validator import validate_citations
from citation_handler import get_source_pages


# =========================================================
# 1. LOAD EVALUATION QUESTIONS
# =========================================================

questions_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/evaluation/questions.json"

with open(questions_path, "r", encoding="utf-8") as file:
    questions = json.load(file)


# =========================================================
# 2. EVALUATION FUNCTION
# =========================================================

def evaluate_result(result, expected_page):
    """
    Evaluate the citations returned by the RAG pipeline.
    """

    citation_ids = result["citation_ids"]
    final_context = result["final_context"]

    # Validate citations again independently
    validated_documents = validate_citations(
        citation_ids,
        final_context
    )

    source_pages = get_source_pages(
        validated_documents
    )

    # Check that every LLM citation is valid
    citation_valid = (
        len(citation_ids) > 0
        and len(validated_documents) == len(citation_ids)
    )

    # Check whether expected page appears in sources
    if isinstance(expected_page, list):
        expected_page_found = any(
            page in source_pages
            for page in expected_page
        )
    else:
        expected_page_found = expected_page in source_pages

    return {
        "citation_valid": citation_valid,
        "expected_page_found": expected_page_found,
        "source_pages": source_pages,
        "citation_count": len(citation_ids),
        "valid_citation_count": len(validated_documents)
    }


# =========================================================
# 3. RUN EVALUATION
# =========================================================

print("\n")
print("======================================")
print("       10-QUESTION CITATION EVALUATION")
print("======================================")


citation_valid_count = 0
expected_page_count = 0


for index, item in enumerate(questions, start=1):

    question = item["question"]
    expected_page = item["expected_page"]

    print("\n")
    print("--------------------------------------")
    print(f"Question {index}/10")
    print("--------------------------------------")

    print("Question:", question)
    print("Expected page:", expected_page)

    # Run complete RAG pipeline
    result = run_rag(question)

    if result.get("error"):

        print("\nRAG ERROR:")
        print(result["error"])

        print("\nStatus: ERROR")

        continue

    # Evaluate result
    evaluation = evaluate_result(
        result,
        expected_page
    )

    # Count metrics
    if evaluation["citation_valid"]:
        citation_valid_count += 1

    if evaluation["expected_page_found"]:
        expected_page_count += 1

    # Display result
    print("\nAnswer:")
    print(result["answer"])

    print("\nLLM citations:")
    print(result["citation_ids"])
    
    print("\nSource pages:")
    print(evaluation["source_pages"])

    print(
        "\nCitation validity:",
        "PASS" if evaluation["citation_valid"] else "FAIL"
    )

    print(
        "Expected page coverage:",
        "PASS" if evaluation["expected_page_found"] else "FAIL"
    )


# =========================================================
# 4. FINAL SUMMARY
# =========================================================

total_questions = len(questions)

print("\n")
print("======================================")
print("          EVALUATION SUMMARY")
print("======================================")

print(
    f"Citation validity: "
    f"{citation_valid_count}/{total_questions}"
)

print(
    f"Expected page coverage: "
    f"{expected_page_count}/{total_questions}"
)

print(
    f"Citation validity percentage: "
    f"{(citation_valid_count / total_questions) * 100:.1f}%"
)

print(
    f"Expected page coverage percentage: "
    f"{(expected_page_count / total_questions) * 100:.1f}%"
)

print("======================================")