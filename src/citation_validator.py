def build_chunk_lookup(documents):
    """
    Create a lookup from chunk_id to Document.
    """
    return {
        document.metadata["chunk_id"]: document
        for document in documents
        if "chunk_id" in document.metadata
    }


def validate_citations(citation_ids, documents):
    """
    Validate LLM-generated citation IDs against the provided documents.
    """
    chunk_lookup = build_chunk_lookup(documents)

    valid_documents = []

    for citation_id in citation_ids:
        if not citation_id.startswith("CHUNK_"):
            continue

        try:
            chunk_id = int(citation_id.replace("CHUNK_", ""))
        except ValueError:
            continue

        if chunk_id in chunk_lookup:
            valid_documents.append(chunk_lookup[chunk_id])

    return valid_documents

if __name__ == "__main__":
    class MockDocument:
        def __init__(self, chunk_id, page):
            self.metadata = {
                "chunk_id": chunk_id,
                "pdf_page": page
            }

    test_documents = [
        MockDocument(31, 10),
        MockDocument(105, 25),
        MockDocument(106, 25)
    ]

    test_citations = [
        "CHUNK_31",
        "CHUNK_105",
        "CHUNK_999"
    ]

    validated = validate_citations(
        test_citations,
        test_documents
    )

    print("Requested citations:", test_citations)

    print(
        "Validated citations:",
        [
            f"CHUNK_{doc.metadata['chunk_id']}"
            for doc in validated
        ]
    )

    print(
        "Validated pages:",
        [
            doc.metadata["pdf_page"]
            for doc in validated
        ]
    )