import json
import re


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

input_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_cleaned_pages.json"

output_path = "C:/Users/Admin/Desktop/FinancialResearchRAG/data/processed/infosys_chunks.json"


# --------------------------------------------------
# 2. Chunk settings
# --------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# 3. Load cleaned pages
# --------------------------------------------------

with open(input_path, "r", encoding="utf-8") as file:
    pages = json.load(file)


print("Pages loaded:", len(pages))


# --------------------------------------------------
# 4. Function to split text into sentences
# --------------------------------------------------

def split_into_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# --------------------------------------------------
# 5. Create chunks
# --------------------------------------------------

chunks = []


for page in pages:

    text = page["text"]
    pdf_page = page["pdf_page"]

    sentences = split_into_sentences(text)

    current_sentences = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        # ------------------------------------------
        # Check whether adding this sentence
        # exceeds our target chunk size
        # ------------------------------------------

        if (
            current_sentences
            and current_length + sentence_length > CHUNK_SIZE
        ):

            chunk_text = " ".join(current_sentences)

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "company": "Infosys",
                    "document": "Infosys Integrated Annual Report 2025-26",
                    "financial_year": "2025-26",
                    "pdf_page": pdf_page
                }
            })

            # --------------------------------------
            # Create overlap using previous sentences
            # --------------------------------------

            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(current_sentences):

                if overlap_length + len(previous_sentence) > CHUNK_OVERLAP:
                    break

                overlap_sentences.insert(
                    0,
                    previous_sentence
                )

                overlap_length += len(previous_sentence)

            current_sentences = overlap_sentences

            current_length = overlap_length

        # ------------------------------------------
        # Add new sentence
        # ------------------------------------------

        current_sentences.append(sentence)

        current_length += sentence_length + 1


    # --------------------------------------------------
    # Add remaining sentences from the page
    # --------------------------------------------------

    if current_sentences:

        chunk_text = " ".join(current_sentences)

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "company": "Infosys",
                "document": "Infosys Integrated Annual Report 2025-26",
                "financial_year": "2025-26",
                "pdf_page": pdf_page
            }
        })


# --------------------------------------------------
# 6. Save chunks
# --------------------------------------------------

with open(output_path, "w", encoding="utf-8") as file:

    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=4
    )


# --------------------------------------------------
# 7. Display information
# --------------------------------------------------

print("Total chunks:", len(chunks))

print("Saved chunks to:", output_path)


# --------------------------------------------------
# 8. Display first chunk
# --------------------------------------------------

print("\n----- FIRST CHUNK -----\n")

print(chunks[0]["text"])


print("\n----- FIRST CHUNK LENGTH -----\n")

print(len(chunks[0]["text"]))


print("\n----- FIRST CHUNK METADATA -----\n")

print(chunks[0]["metadata"])


# --------------------------------------------------
# 9. Display second chunk
# --------------------------------------------------

print("\n----- SECOND CHUNK -----\n")

print(chunks[1]["text"])


print("\n----- SECOND CHUNK LENGTH -----\n")

print(len(chunks[1]["text"]))