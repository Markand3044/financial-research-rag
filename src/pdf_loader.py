from pypdf import PdfReader


def extract_pages(pdf_path):
    """
    Extract text from every page of a PDF.

    Parameters:
        pdf_path: Path to the PDF file.

    Returns:
        List of dictionaries containing PDF page number and text.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        pages.append({
            "pdf_page": page_number,
            "text": text
        })

    return pages