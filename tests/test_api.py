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

    assert isinstance(
        data["documents"],
        list
    )


def test_invalid_document():
    response = client.get(
        "/documents/document-that-does-not-exist"
    )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "Document not found."
    }