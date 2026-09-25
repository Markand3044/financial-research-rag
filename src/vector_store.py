from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_faiss_vectorstore(chunks, vectorstore_path):
    """
    Create and save a FAISS vector store from chunks.

    Parameters:
        chunks: List of chunk dictionaries.
        vectorstore_path: Path where the FAISS index will be saved.

    Returns:
        FAISS vector store.
    """

    documents = []

    for chunk in chunks:

        document = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )

        documents.append(document)

    print("Documents created:", len(documents))

    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )

    print("FAISS vector store created")

    vector_store.save_local(vectorstore_path)

    print("FAISS vector store saved:", vectorstore_path)

    return vector_store