def get_source_pages(documents):
    """
    Extract unique PDF page numbers from retrieved documents.
    """

    pages = sorted({
        document.metadata["pdf_page"]
        for document in documents
        if "pdf_page" in document.metadata
    })

    return pages


def format_sources(pages):
    """
    Format PDF page numbers for displaying to the user.
    """

    if not pages:
        return "No source pages available."

    return ", ".join(
        f"PDF Page {page}"
        for page in pages
    )

if __name__ == "__main__":

    class MockDocument:

        def __init__(self, page):
            self.metadata = {
                "pdf_page": page
            }


    test_documents = [
        MockDocument(10),
        MockDocument(25),
        MockDocument(10),
        MockDocument(269)
    ]

    pages = get_source_pages(test_documents)

    print("Pages:", pages)
    print("Sources:", format_sources(pages))