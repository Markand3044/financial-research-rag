from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


def create_chunks(pages, company, document, financial_year):
    """
    Convert cleaned PDF pages into LangChain text chunks.

    Parameters:
        pages: Cleaned pages from clean_pages()
        company: Company name
        document: Document name
        financial_year: Financial year

    Returns:
        List of chunks with text and metadata.
    """

    chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = text_splitter.split_text(
            page["text"]
        )

        for chunk_text in page_chunks:

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "company": company,
                    "document": document,
                    "financial_year": financial_year,
                    "pdf_page": page["pdf_page"],
                    "chunk_id": chunk_id
                }
            })

            chunk_id += 1

    return chunks