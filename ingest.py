from rag.document_loader import load_pdfs, split_documents
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
def main():
    print("=" * 60)
    print("STARTING DOCUMENT INGESTION")
    print("=" * 60)
    # 1. Load PDFs
    documents = load_pdfs()
    # 2. Split documents into chunks
    chunks = split_documents(documents, chunk_size=1000,chunk_overlap=200)
    # 3. Create embedding manager
    embedding_manager = EmbeddingManager()
    # 4. Extract text from chunks
    texts = [doc.page_content for doc in chunks]
    # 5. Generate embeddings
    embeddings = (embedding_manager.generate_embeddings(texts))
    # 6. Create vector store
    vector_store = VectorStore()
    # 7. Store documents + embeddings
    vector_store.add_documents(chunks, embeddings)

    print("=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Total chunks stored: "
        f"{vector_store.collection.count()}"
    )

if __name__ == "__main__":
    main()