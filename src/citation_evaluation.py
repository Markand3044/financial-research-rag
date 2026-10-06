import json
import os
import sys
import re

from sentence_transformers import SentenceTransformer, util

# Allow imports when running:
# python src/citation_evaluation.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reg_pipeline_v2 import run_rag


# =========================================================
# 1. CONFIGURATION
# =========================================================

QUESTIONS_PATH = (
    "data/evaluation/questions.json"
)

VECTORSTORE_PATH = (
    "data/vectorstore/infosys_faiss"
)

ANSWER_SIMILARITY_THRESHOLD = 0.70


# =========================================================
# 2. LOAD EVALUATION QUESTIONS
# =========================================================

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(file)


# =========================================================
# 3. LOAD SEMANTIC SIMILARITY MODEL
# =========================================================

print("Loading answer evaluation model...")

similarity_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Answer evaluation model loaded.")


# =========================================================
# 4. ANSWER SIMILARITY FUNCTION
# =========================================================

def calculate_fact_match(generated_answer, reference_answer):
    """
    Check whether important factual values from the reference
    answer are present in the generated answer.
    """

    if not reference_answer:
        return None

    reference_facts = extract_key_facts(reference_answer)
    generated_facts = extract_key_facts(generated_answer)

    if not reference_facts:
        return None

    matched_facts = reference_facts.intersection(generated_facts)

    return len(matched_facts) / len(reference_facts)

def calculate_answer_similarity(
    generated_answer,
    reference_answer
):
    """
    Calculate semantic similarity between the
    generated answer and the reference answer.

    Returns a value between 0 and 1.
    """

    if not reference_answer:
        return None

    generated_embedding = similarity_model.encode(
        generated_answer,
        convert_to_tensor=True
    )

    reference_embedding = similarity_model.encode(
        reference_answer,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        generated_embedding,
        reference_embedding
    ).item()

    return similarity

def extract_key_facts(text):
    """
    Extract important factual values from an answer.

    Handles:
    - percentages
    - currency values
    - numbers
    """

    if not text:
        return set()

    text = text.lower()

    facts = set()

    # --------------------------------------------------
    # 1. Percentages
    # --------------------------------------------------
    percentages = re.findall(
        r"\d+(?:\.\d+)?\s*%",
        text
    )

    for value in percentages:
        facts.add(
            value.replace(" ", "")
        )

    # --------------------------------------------------
    # 2. Currency values
    # --------------------------------------------------
    currency_values = re.findall(
        r"(?:₹|rs\.?|inr)\s*[\d,]+(?:\.\d+)?"
        r"(?:\s*(?:crore|lakh|million|billion))?",
        text
    )

    for value in currency_values:
        normalized = re.sub(
            r"\s+",
            " ",
            value.strip()
        )
        facts.add(normalized)

    # --------------------------------------------------
    # 3. Plain numbers
    # --------------------------------------------------
    numbers = re.findall(
        r"\b\d[\d,]*(?:\.\d+)?\b",
        text
    )

    for value in numbers:

        # Ignore years such as 2025 and 2026.
        if len(value) == 4 and value.startswith(("19", "20")):
            continue

        facts.add(value)

    return facts

# =========================================================
# 5. EVALUATION FUNCTION
# =========================================================

def evaluate_result(
    result,
    item
):
    """
    Evaluate one RAG result.

    Checks:

    1. Citation validity
    2. Expected source-page coverage
    3. Correct abstention
    4. Semantic similarity with reference answer
    """

    answer = result.answer
    citation_ids = result.citations
    source_pages = result.source_pages

    expected_page = item["expected_page"]
    answerable = item.get("answerable", True)
    reference_answer = item.get("reference_answer")

    # -----------------------------------------------------
    # Expected page coverage
    # -----------------------------------------------------

    if answerable:

        if isinstance(expected_page, list):

            expected_page_found = any(
                page in source_pages
                for page in expected_page
            )

        else:

            expected_page_found = (
                expected_page in source_pages
            )

    else:

        expected_page_found = None

    # -----------------------------------------------------
    # Citation validity
    #
    # run_rag() already validates citations before
    # returning RAGResult.
    #
    # Therefore, citations returned by RAGResult are
    # considered validated citations.
    # -----------------------------------------------------

    if answerable:

        citation_valid = len(citation_ids) > 0

    else:

        citation_valid = len(citation_ids) == 0

    # -----------------------------------------------------
    # Abstention
    # -----------------------------------------------------

    if not answerable:

        answer_lower = answer.lower()

        abstention_correct = (
            "does not contain enough information"
            in answer_lower
            or "not enough information"
            in answer_lower
            or "cannot determine"
            in answer_lower
            or "cannot be determined"
            in answer_lower
        )

    else:

        abstention_correct = None

    # -----------------------------------------------------
    # Answer similarity
    # -----------------------------------------------------

    if answerable and reference_answer:

        answer_similarity = calculate_answer_similarity(
            answer,
            reference_answer
        )

        fact_match = calculate_fact_match(
            answer,
            reference_answer
        )

        # Answer passes if either:
        # 1. semantic similarity is high
        # OR
        # 2. important factual values match

        answer_similarity_pass = (
            answer_similarity >= ANSWER_SIMILARITY_THRESHOLD
            or fact_match == 1.0
        )

    else:
        answer_similarity = None
        fact_match = None
        answer_similarity_pass = None
    # -----------------------------------------------------
    # Return evaluation
    # -----------------------------------------------------

    return {
        "citation_valid": citation_valid,
        "expected_page_found": expected_page_found,
        "abstention_correct": abstention_correct,
        "answer_similarity": answer_similarity,
        "fact_match": fact_match,
        "answer_similarity_pass": answer_similarity_pass,
        "source_pages": source_pages,
        "citation_count": len(citation_ids)
    }


# =========================================================
# 6. EVALUATION COUNTERS
# =========================================================

citation_valid_count = 0
expected_page_count = 0
abstention_correct_count = 0
answer_similarity_pass_count = 0

answerable_count = 0
unanswerable_count = 0


# =========================================================
# 7. RUN EVALUATION
# =========================================================

print("\n")
print("==============================================")
print("       RAG ANSWER & CITATION EVALUATION")
print("==============================================")

print(
    f"\nTotal questions: {len(questions)}"
)

print(
    f"Answer similarity threshold: "
    f"{ANSWER_SIMILARITY_THRESHOLD}"
)


for index, item in enumerate(
    questions,
    start=1
):

    question = item["question"]
    expected_page = item["expected_page"]
    answerable = item.get(
        "answerable",
        True
    )

    reference_answer = item.get(
        "reference_answer"
    )

    print("\n")
    print("----------------------------------------------")
    print(
        f"Question {index}/{len(questions)}"
    )
    print("----------------------------------------------")

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
    # Run RAG
    # -----------------------------------------------------

    try:

        result = run_rag(
            question,
            VECTORSTORE_PATH
        )

    except Exception as error:

        print("\nRAG ERROR:")
        print(error)

        print("\nStatus: ERROR")

        continue

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    evaluation = evaluate_result(
        result,
        item
    )

    # -----------------------------------------------------
    # Citation validity
    # -----------------------------------------------------

    if evaluation["citation_valid"]:

        citation_valid_count += 1

    # -----------------------------------------------------
    # Expected page
    # -----------------------------------------------------

    if (
        answerable
        and evaluation["expected_page_found"]
    ):

        expected_page_count += 1

    # -----------------------------------------------------
    # Abstention
    # -----------------------------------------------------

    if (
        not answerable
        and evaluation["abstention_correct"]
    ):

        abstention_correct_count += 1

    # -----------------------------------------------------
    # Answer similarity
    # -----------------------------------------------------

    if (
        answerable
        and evaluation["answer_similarity_pass"]
    ):

        answer_similarity_pass_count += 1

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print("\nGenerated answer:")
    print(result.answer)

    if reference_answer:

        print("\nReference answer:")
        print(reference_answer)

    print("\nLLM citations:")
    print(result.citations)

    print("\nSource pages:")
    print(result.source_pages)

    # -----------------------------------------------------
    # Citation result
    # -----------------------------------------------------

    print(
        "\nCitation validity:",
        "PASS"
        if evaluation["citation_valid"]
        else "FAIL"
    )

    # -----------------------------------------------------
    # Page coverage
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
            "Expected page coverage: N/A"
        )

    # -----------------------------------------------------
    # Answer similarity
    # -----------------------------------------------------

    if answerable:

        print(
            "Answer similarity:",
            f"{evaluation['answer_similarity']:.3f}"
        )

        if evaluation["fact_match"] is not None:
            print(
                "Fact match:",
                f"{evaluation['fact_match']:.3f}"
            )
        else:
            print("Fact match: N/A")

        print(
            "Answer similarity result:",
            "PASS"
            if evaluation["answer_similarity_pass"]
            else "FAIL"
        )

    # -----------------------------------------------------
    # Abstention
    # -----------------------------------------------------

    if not answerable:

        print(
            "Abstention:",
            "PASS"
            if evaluation["abstention_correct"]
            else "FAIL"
        )


# =========================================================
# 8. FINAL SUMMARY
# =========================================================

total_questions = len(questions)

print("\n")
print("==============================================")
print("              EVALUATION SUMMARY")
print("==============================================")


# ---------------------------------------------------------
# Citation validity
# ---------------------------------------------------------

print(
    f"\nOverall citation validity: "
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

    print(
        f"\nAnswer similarity passing: "
        f"{answer_similarity_pass_count}/"
        f"{answerable_count}"
    )

    print(
        f"Answer similarity passing percentage: "
        f"{(answer_similarity_pass_count / answerable_count) * 100:.1f}%"
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
        f"{abstention_correct_count}/"
        f"{unanswerable_count}"
    )

    print(
        f"Correct abstention percentage: "
        f"{(abstention_correct_count / unanswerable_count) * 100:.1f}%"
    )


print("\n==============================================")