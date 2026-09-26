from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from pathlib import Path
import shutil

from src.reg_pipeline_v2 import run_rag
from src.pdf_loader import extract_pages
from src.text_cleaner import clean_pages
from src.chunker_by_langchain import create_chunks
from src.vector_store import create_faiss_vectorstore
from src.document_registry import (register_document, get_document, load_registry)


app = FastAPI(
    title="Financial Research RAG Assistant",
    description="API for asking questions about financial reports",
    version="1.0.0"
)

UPLOAD_DIR = Path("C:/Users/Admin/Desktop/FinancialResearchRAG/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str
    document_id: str


@app.get("/")
def root():
    return {
        "message": "Financial Research RAG Assistant API is running"
    }

@app.get("/documents")
def get_documents():
    registry = load_registry()

    documents = []

    for document_id, document in registry.items():
        documents.append({
            "document_id": document_id,
            "filename": document["filename"]
        })

    return {
        "documents": documents
    }

@app.get("/documents/{document_id}")
def get_document_details(document_id: str):
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "vectorstore_path": document["vectorstore_path"]
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file_path.stat().st_size == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    document_id = Path(file.filename).stem

    if get_document(document_id) is not None:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail="A document with this filename already exists."
        )

    try:
        # ---------------------------------------------
        # 1. Extract PDF text
        # ---------------------------------------------

        pages = extract_pages(file_path)

        # ---------------------------------------------
        # 2. Clean extracted text
        # ---------------------------------------------

        cleaned_pages = clean_pages(pages)

        if not cleaned_pages:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in the PDF."
            )

        # ---------------------------------------------
        # 3. Create chunks
        # ---------------------------------------------

        chunks = create_chunks(
            cleaned_pages,
            company=document_id,
            document=file.filename,
            financial_year="unknown"
        )

        # ---------------------------------------------
        # 4. Create FAISS vector store
        # ---------------------------------------------

        vectorstore_path = (
            Path("data/vectorstore")
            / f"{document_id}_faiss"
        )

        create_faiss_vectorstore(
            chunks,
            vectorstore_path
        )

        # ---------------------------------------------
        # 5. Register document
        # ---------------------------------------------

        register_document(
            document_id=document_id,
            filename=file.filename,
            vectorstore_path=vectorstore_path
        )

    except HTTPException:
        raise

    except Exception as e:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )

    return {
        "message": "PDF uploaded and processed successfully",
        "filename": file.filename,
        "total_pages": len(pages),
        "cleaned_pages": len(cleaned_pages),
        "total_chunks": len(chunks),
        "document_id": document_id
    }

@app.post("/ask")
def ask_question(request: QuestionRequest):

    document = get_document(request.document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    vectorstore_path = document["vectorstore_path"]

    result = run_rag(
        request.question,
        str(vectorstore_path)
    )

    return result


