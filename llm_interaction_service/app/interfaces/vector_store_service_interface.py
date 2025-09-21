from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document

class VectorStoreServiceInterface(ABC):
    """Interface for vector store services."""

    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: List[dict]):
        """Add texts to the vector store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> List[Document]:
        """Search for similar texts in the vector store."""
        pass