import json

from reg_pipeline_v2 import run_rag
from citation_validator import validate_citations
from citation_handler import get_source_pages


# =========================================================
# 1. LOAD EVALUATION QUESTIONS
# =========================================================

questions_path = (
    "C:/Users/Admin/Desktop/FinancialResearchRAG/"
    "data/evaluation/questions.json"
)

with open(questions_path, "r", encoding="utf-8") as file:
    questions = json.load(file)


# =========================================================
# 2. EVALUATION FUNCTION
# =========================================================

def evaluate_result(result, expected_page):
    """
    Evaluate the RAG result.

    For answerable questions:
        - citations must be valid
        - expected source page must be found

    For unanswerable questions:
        - citations should be empty
        - RAG should abstain from answering
    """

    citation_ids = result["citation_ids"]
    final_context = result["final_context"]

    # -----------------------------------------------------
    # Validate citations independently
    # -----------------------------------------------------

    validated_documents = validate_citations(
        citation_ids,
        final_context
    )

    source_pages = get_source_pages(
        validated_documents
    )

    # -----------------------------------------------------
    # Determine whether question is answerable
    # -----------------------------------------------------

    answerable = expected_page is not None

    # -----------------------------------------------------
    # Answerable question
    # -----------------------------------------------------

    if answerable:

        # Answerable questions must have valid citations
        citation_valid = (
            len(citation_ids) > 0
            and len(validated_documents) == len(citation_ids)
        )

        # Check expected page
        if isinstance(expected_page, list):

            expected_page_found = any(
                page in source_pages
                for page in expected_page
            )

        else:

            expected_page_found = (
                expected_page in source_pages
            )

        # Not applicable for answerable questions
        abstention_correct = None

    # -----------------------------------------------------
    # Unanswerable question
    # -----------------------------------------------------

    else:

        # Unanswerable questions should have NO citations
        citation_valid = (
            len(citation_ids) == 0
            and len(validated_documents) == 0
        )

        # There is no expected source page
        expected_page_found = True

        # Check whether the RAG correctly abstained
        answer_lower = result["answer"].lower()

        abstention_correct = (
            "does not contain enough information"
            in answer_lower
            or "not enough information"
            in answer_lower
            or "cannot determine"
            in answer_lower
        )

    # -----------------------------------------------------
    # Return evaluation result
    # IMPORTANT:
    # This return must be OUTSIDE the if/else blocks
    # -----------------------------------------------------

    return {
        "citation_valid": citation_valid,
        "expected_page_found": expected_page_found,
        "abstention_correct": abstention_correct,
        "source_pages": source_pages,
        "citation_count": len(citation_ids),
        "valid_citation_count": len(validated_documents)
    }


# =========================================================
# 3. RUN EVALUATION
# =========================================================

print("\n")
print("======================================")
print("   RAG ANSWER & CITATION EVALUATION")
print("======================================")


# ---------------------------------------------------------
# Counters
# ---------------------------------------------------------

citation_valid_count = 0
expected_page_count = 0
abstention_correct_count = 0

answerable_count = 0
unanswerable_count = 0


# =========================================================
# 4. EVALUATE EACH QUESTION
# =========================================================

for index, item in enumerate(questions, start=1):

    question = item["question"]
    expected_page = item["expected_page"]

    # If answerable field exists, use it.
    # Otherwise assume question is answerable.
    answerable = item.get("answerable", True)

    print("\n")
    print("--------------------------------------")
    print(
        f"Question {index}/{len(questions)}"
    )
    print("--------------------------------------")

    print("Question:", question)
    print("Expected page:", expected_page)
    print("Answerable:", answerable)

    # -----------------------------------------------------
    # Count question types
    # -----------------------------------------------------

    if answerable:
        answerable_count += 1
    else:
        unanswerable_count += 1

    # -----------------------------------------------------
    # Run complete RAG pipeline
    # -----------------------------------------------------

    result = run_rag(question)

    if result.get("error"):

        print("\nRAG ERROR:")
        print(result["error"])

        print("\nStatus: ERROR")

        continue

    # -----------------------------------------------------
    # Evaluate result
    # -----------------------------------------------------

    evaluation = evaluate_result(
        result,
        expected_page
    )

    # -----------------------------------------------------
    # Count citation validity
    # -----------------------------------------------------

    if evaluation["citation_valid"]:
        citation_valid_count += 1

    # -----------------------------------------------------
    # Count expected page coverage
    # -----------------------------------------------------

    if answerable and evaluation["expected_page_found"]:
        expected_page_count += 1

    # -----------------------------------------------------
    # Count correct abstention
    # -----------------------------------------------------

    if (
        not answerable
        and evaluation["abstention_correct"]
    ):
        abstention_correct_count += 1

    # -----------------------------------------------------
    # Display answer
    # -----------------------------------------------------

    print("\nAnswer:")
    print(result["answer"])

    print("\nLLM citations:")
    print(result["citation_ids"])

    print("\nSource pages:")
    print(evaluation["source_pages"])

    # -----------------------------------------------------
    # Citation validity
    # -----------------------------------------------------

    print(
        "\nCitation validity:",
        "PASS"
        if evaluation["citation_valid"]
        else "FAIL"
    )

    # -----------------------------------------------------
    # Expected page coverage
    # -----------------------------------------------------

    if answerable:

        print(
            "Expected page coverage:",
            "PASS"
            if evaluation["expected_page_found"]
            else "FAIL"
        )

    else:

        print(
            "Expected page coverage:",
            "N/A"
        )

    # -----------------------------------------------------
    # Abstention result
    # -----------------------------------------------------

    if not answerable:

        print(
            "Abstention:",
            "PASS"
            if evaluation["abstention_correct"]
            else "FAIL"
        )


# =========================================================
# 5. FINAL SUMMARY
# =========================================================

total_questions = len(questions)

print("\n")
print("======================================")
print("          EVALUATION SUMMARY")
print("======================================")


# ---------------------------------------------------------
# Overall citation validity
# ---------------------------------------------------------

print(
    f"Overall citation validity: "
    f"{citation_valid_count}/{total_questions}"
)

print(
    f"Overall citation validity percentage: "
    f"{(citation_valid_count / total_questions) * 100:.1f}%"
)


# ---------------------------------------------------------
# Answerable questions
# ---------------------------------------------------------

if answerable_count > 0:

    print(
        f"\nAnswerable questions: "
        f"{answerable_count}"
    )

    print(
        f"Expected page coverage: "
        f"{expected_page_count}/{answerable_count}"
    )

    print(
        f"Expected page coverage percentage: "
        f"{(expected_page_count / answerable_count) * 100:.1f}%"
    )


# ---------------------------------------------------------
# Unanswerable questions
# ---------------------------------------------------------

if unanswerable_count > 0:

    print(
        f"\nUnanswerable questions: "
        f"{unanswerable_count}"
    )

    print(
        f"Correct abstention: "
        f"{abstention_correct_count}/{unanswerable_count}"
    )

    print(
        f"Correct abstention percentage: "
        f"{(abstention_correct_count / unanswerable_count) * 100:.1f}%"
    )


print("======================================")