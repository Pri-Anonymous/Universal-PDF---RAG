from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Project root = parent of the "rag" folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIRECTORY = PROJECT_ROOT / "data" / "pdf_files"


def load_pdfs() -> List[Document]:
    """
    Load all PDF files from the data/pdf_files directory.
    """

    if not PDF_DIRECTORY.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIRECTORY}"
        )

    loader = DirectoryLoader(
        str(PDF_DIRECTORY),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} PDF pages.")

    return documents


def split_documents(documents: List[Document],chunk_size: int = 1000,chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for RAG.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into "
        f"{len(split_docs)} chunks.")

    if split_docs:
        print("\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs