from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document

class VectorStoreServiceInterface(ABC):
    """Interface for vector store services."""

    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: List[dict], ids: List[str] = None):
        """Add texts to the vector store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> List[Document]:
        """Search for similar texts in the vector store."""
        pass

    @abstractmethod
    def get_collection(self, collection_name: str) -> dict:
        """Get all texts from a collection."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete all texts from a collection."""
        pass
    
    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the current collection."""
        pass
    
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        pass