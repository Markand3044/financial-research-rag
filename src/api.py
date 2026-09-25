from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from pathlib import Path
import shutil

from src.reg_pipeline_v2 import run_rag
from src.pdf_loader import extract_pages
from src.text_cleaner import clean_pages
from src.chunker_by_langchain import create_chunks
from src.vector_store import create_faiss_vectorstore


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

    # ---------------------------------------------
    # 1. Extract PDF pages
    # ---------------------------------------------

    pages = extract_pages(file_path)

    # ---------------------------------------------
    # 2. Clean extracted text
    # ---------------------------------------------

    cleaned_pages = clean_pages(pages)

    # ---------------------------------------------
    # 3. Create chunks
    # ---------------------------------------------

    chunks = create_chunks(
        cleaned_pages,
        company="Infosys",
        document="Infosys Integrated Annual Report 2025-2026",
        financial_year="2025-2026"
    )

    # ---------------------------------------------
    # 4. Create FAISS vector store
    # ---------------------------------------------

    document_id = Path(file.filename).stem

    vectorstore_path = (
        Path("data/vectorstore")
        / f"{document_id}_faiss"
    )

    create_faiss_vectorstore(
        chunks,
        vectorstore_path
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

    vectorstore_path = (
        Path("data/vectorstore")
        / f"{request.document_id}_faiss"
    )

    if not vectorstore_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document vector store not found."
        )

    result = run_rag(
        request.question,
        str(vectorstore_path)
    )

    return result


