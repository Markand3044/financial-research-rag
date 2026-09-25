import re


def clean_pages(pages):

    cleaned_pages = []

    for page in pages:

        text = page["text"]

        if text is None:
            text = ""

        text = text.replace("\u00ad", "")

        text = re.sub(
            r"(\w)-\s*\n\s*(\w)",
            r"\1-\2",
            text
        )

        text = re.sub(
            r"\s*\n\s*",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        text = text.strip()

        if text:

            cleaned_pages.append({
                "pdf_page": page["pdf_page"],
                "text": text
            })

    return cleaned_pages