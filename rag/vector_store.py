import chromadb
import os
from pathlib import Path
import uuid
import numpy as np
from typing import List,Any

class VectorStore:
    '''Manages a vector space for document embeddings using ChromaDB.'''

    def __init__(self,collection_name:str = "pdf_files",persist_directory:str | None = None):
        '''
        Initialize the VectorStore with a collection name and persistence directory.
        
        Args:
            collection_name (str): Name of the ChromaDB collection.
            persist_directory (str): Directory to persist the ChromaDB data.
        '''
        #Find the root of directory of the project
        PROJECT_ROOT = Path(__file__).resolve().parent.parent

        #If no persistence directory is provided, use chromadb directory in the data folder of the project  
            
        if persist_directory is None:
            persist_directory = str(PROJECT_ROOT / "data" / "chromadb")

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        "Initialize the ChromaDB client and collection."
        ##Deleting the old collection if it exists

        try:
            #Creating perisistent chromadb client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            #Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space":"cosine","description": "Collection of document embeddings"})
            print(f"ChromaDB collection '{self.collection_name}' initialized successfully.")
            print(f"Existing documents in the collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB collection '{self.collection_name}': {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        '''
        Add documents and their embeddings to the ChromaDB collection.
        
        Args:
            documents (List[Any]): List of documents to add.
            embeddings (np.ndarray): Corresponding embeddings for the documents.
        '''
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        print(f"Adding {len(documents)} documents to the collection '{self.collection_name}'...")
        #prepare the data for ChromaDB
        ids =[]
        metadatas = []
        document_texts = []
        embeddings_list = []

        for i,(doc, embedding) in enumerate(zip(documents, embeddings)):
           #Generate a unique ID for each document
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
          
          #Generate metadata for each document
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

           #Document text
            document_texts.append(doc.page_content)

           #embedding
            embeddings_list.append(embedding.tolist())
        
        try:
            #Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=document_texts,
                embeddings=embeddings_list)
            print(f"Documents added successfully. Total documents in collection: {self.collection.count()}")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to ChromaDB collection '{self.collection_name}': {e}")
            raise
