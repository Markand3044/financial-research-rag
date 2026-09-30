from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Financial Research RAG Assistant API is running"
    }


def test_get_documents():
    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_invalid_document():
    response = client.get(
        "/documents/document-that-does-not-exist"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Document not found."
    }


def test_upload_rejects_non_pdf():

    response = client.post(
        "/upload",
        files={
            "file": (
                "test.txt",
                b"This is not a PDF.",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Only PDF files are allowed."
    }


@patch("src.api.get_document")
def test_upload_rejects_duplicate(mock_get_document):

    mock_get_document.return_value = {
        "filename": "existing.pdf",
        "stored_filename": "existing.pdf",
        "vectorstore_path": "data/vectorstore/existing_faiss"
    }

    response = client.post(
        "/upload",
        files={
            "file": (
                "existing.pdf",
                b"fake pdf content",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 409

    assert response.json() == {
        "detail": "A document with this filename already exists."
    }