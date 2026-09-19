from fastapi import FastAPI
from pydantic import BaseModel

from src.reg_pipeline_v2 import run_rag


app = FastAPI(
    title="Financial Research RAG Assistant",
    description="API for asking questions about financial reports",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Financial Research RAG Assistant API is running"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = run_rag(request.question)

    return result