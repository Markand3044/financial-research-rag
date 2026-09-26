import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Financial Research RAG Assistant",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Financial Research RAG Assistant")

st.write(
    "Upload a financial report and ask questions about its contents."
)


# ============================================================
# DOCUMENTS
# ============================================================

st.header("Select Financial Report")


try:
    response = requests.get(
        f"{API_URL}/documents"
    )

    if response.status_code == 200:

        documents = response.json()["documents"]

        if documents:

            document_options = {
                document["filename"]: document["document_id"]
                for document in documents
            }

            selected_filename = st.selectbox(
                "Available reports",
                list(document_options.keys())
            )

            selected_document_id = document_options[
                selected_filename
            ]

            st.success(
                f"Selected: {selected_filename}"
            )

        else:

            st.info(
                "No financial reports have been uploaded yet."
            )

    else:

        st.error(
            "Could not load documents from the API."
        )

except requests.exceptions.ConnectionError:

    st.error(
        "FastAPI server is not running. "
        "Please start it with: uvicorn src.api:app --reload"
    )


st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.header("Upload Financial Report")


uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    st.success(
        f"Selected file: {uploaded_file.name}"
    )


st.divider()


# ============================================================
# ASK QUESTION
# ============================================================

st.header("Ask a Question")


question = st.text_input(
    "Enter your question"
)


if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif "selected_document_id" not in locals():

        st.warning(
            "Please select a financial report first."
        )

    else:

        try:

            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "document_id": selected_document_id
                }
            )

            if response.status_code == 200:

                result = response.json()

                st.subheader("Answer")

                st.write(
                    result["answer"]
                )

                st.subheader("Sources")

                if result["source_pages"]:

                    for page in result["source_pages"]:

                        st.write(
                            f"📄 PDF Page {page}"
                        )

                else:

                    st.write(
                        "No source pages available."
                    )

            else:

                error_detail = response.json().get(
                    "detail",
                    "An error occurred."
                )

                st.error(
                    error_detail
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "FastAPI server is not running."
            )