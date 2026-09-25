from pydantic import BaseModel, Field

class RAGResponse(BaseModel):

    answer: str = Field(
        description="Answer generated from the retrieved context"
    )

    citations: list[str] = Field(
        description="Chunk IDs supporting the answer"
    )

class RAGResult(BaseModel):
    answer: str 
    citations: list[str]
    source_pages: list[int]