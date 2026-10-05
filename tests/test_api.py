from src.output_schema import RAGResult
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

@patch("src.reg_pipeline_v2.run_rag")
def test_ask_returns_rag_result(mock_run_rag):
    mock_run_rag.return_value = RAGResult(
        answer="Infosys had 3,28,594 employees.",
        citations=["CHUNK_31"],
        source_pages=[10]
    )

    response = client.post(
        "/ask",
        json={
            "question": "How many employees did Infosys have?",
            "document_id": "infosys-ar-26"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Infosys had 3,28,594 employees."
    assert data["citations"] == ["CHUNK_31"]
    assert data["source_pages"] == [10]

def test_ask_invalid_document():
    response = client.post(
        "/ask",
        json={
            "question": "How many employees did Infosys have?",
            "document_id": "document-that-does-not-exist"
        }
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found."
    }

@patch("src.api.delete_document")
@patch("src.api.get_document")
def test_delete_document(mock_get_document, mock_delete_document):
    mock_get_document.return_value = {
        "filename": "test.pdf",
        "stored_filename": "test-stored.pdf",
        "vectorstore_path": "data/vectorstore/test_faiss"
    }

    response = client.delete(
        "/documents/test-document"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Document deleted successfully"
    assert data["document_id"] == "test-document"

    mock_delete_document.assert_called_once_with(
        "test-document"
    )