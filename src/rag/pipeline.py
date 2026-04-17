from .loader import load_documents, chunk_text
from .embedder import create_embeddings, retrieve, generate_answer


class RAGPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vector_store = None

    def ingest(self):
        print("Loading documents...")
        docs = load_documents(self.data_path)

        print("Chunking documents...")
        chunks = chunk_text(docs)

        print("Creating embeddings...")
        self.vector_store = create_embeddings(chunks)

        print("RAG ingestion complete.")

    def query(self, question):
        if self.vector_store is None:
            raise ValueError("Run ingest() first!")

        context = retrieve(self.vector_store, question)
        answer = generate_answer(question, context)

        return answer