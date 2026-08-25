# Financial Research RAG Assistant

A Retrieval-Augmented Generation (RAG) system designed to help users
research companies using their financial reports and other corporate
documents.

## Project Goal

The goal of this project is to reduce the time required to read long
financial documents during investment research.

A user will eventually be able to upload a company's financial document
and ask questions about topics such as:

- Revenue and profitability
- Business performance
- Company strategy
- Risks
- Geographic performance
- AI and technology strategy
- Other information available in the uploaded document

The system retrieves relevant information from the document and uses an
LLM to generate an answer with source information.

## Current Architecture

PDF
↓
Text Extraction
↓
Text Cleaning
↓
Document Chunking
↓
Hugging Face Embeddings
↓
FAISS Vector Store
↓
Semantic Retrieval
↓
Qwen 3.6 27B
↓
Answer + Sources

## Current Technologies

- Python
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Groq
- Qwen 3.6 27B
- Git / GitHub

## Current Progress

- [x] Project structure
- [x] PDF text extraction
- [x] Text cleaning
- [x] Document chunking
- [x] Hugging Face embeddings
- [x] FAISS vector store
- [x] Semantic search
- [x] Groq LLM integration
- [x] First end-to-end RAG query
- [ ] RAG evaluation
- [ ] Improved source citations
- [ ] FastAPI backend
- [ ] PDF upload pipeline
- [ ] Streamlit interface
- [ ] Dockerization
- [ ] Final deployment

## Example

Question:

> What was Infosys's total revenue in fiscal 2026?

The system retrieves relevant sections from the annual report and
generates an answer using the retrieved context.

## Disclaimer

This project is intended for educational and research purposes.
It provides information extracted from financial documents and should
not be treated as financial advice or a recommendation to buy or sell
securities.
