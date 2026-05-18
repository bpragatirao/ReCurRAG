"""
Embedding & Retrieval Module for the RAG Pipeline.

Handles:
1. Embedding generation using HuggingFace sentence-transformers (local, free)
2. FAISS vector store creation and similarity search
3. Answer generation via local HuggingFace model (google/flan-t5-base — free)
"""

from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from ..utils.local_llm import answer_question


# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

def get_embeddings_model():
    """
    Returns the local HuggingFace embedding model (free, no API quota).
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Vector Store Creation
# ---------------------------------------------------------------------------

def create_embeddings(chunks: list):
    """
    Create a FAISS vector store from text chunks.

    Args:
        chunks: List of dicts with 'text' and 'source' keys.

    Returns:
        A FAISS vector store instance.
    """
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c.get("chunk_id", i)}
                 for i, c in enumerate(chunks)]

    embeddings_model = get_embeddings_model()

    vector_store = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas)
    return vector_store


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(vector_store, query: str, k: int = 5) -> tuple:
    """
    Retrieve the top-k most relevant chunks from the vector store.

    Args:
        vector_store: A FAISS vector store.
        query: The search query.
        k: Number of top results to retrieve.

    Returns:
        Tuple of (context_string, list_of_source_documents).
    """
    docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get("source", "unknown") for doc in docs]

    return context, sources


# ---------------------------------------------------------------------------
# Generation (Local — Free, no API key needed)
# ---------------------------------------------------------------------------

def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using the local HuggingFace model (google/flan-t5-base).

    Completely free — runs on CPU, no API key or billing required.

    Args:
        query: The user's question.
        context: The retrieved context string.

    Returns:
        The generated answer string.
    """
    try:
        return answer_question(query, context)
    except Exception as e:
        return f"Error generating response: {str(e)}"