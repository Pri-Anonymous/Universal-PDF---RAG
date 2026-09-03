import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingManager:
    """Handles document embedding generation using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the Sentence Transformer model."""

        try:
            print(f"Loading model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            print(
                f"Model loaded successfully. "
                f"Embedding dimension: "
                f"{self.model.get_sentence_embedding_dimension()}"
            )

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, documents: List[str]) -> np.ndarray:

        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before generating embeddings.")

        print(f"Generating embeddings for {len(documents)} documents...")

        embeddings = self.model.encode(
            documents,
            show_progress_bar=True
        )

        print(
            f"Embeddings generated successfully. "
            f"Shape: {embeddings.shape}"
        )

        return embeddings