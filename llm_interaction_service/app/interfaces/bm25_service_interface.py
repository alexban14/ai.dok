from abc import ABC, abstractmethod
from typing import List, Tuple
from langchain.docstore.document import Document

class BM25ServiceInterface(ABC):
    """Interface for BM25 keyword search services."""
    
    @abstractmethod
    def build_index(self, corpus: List[str]) -> None:
        """Build BM25 index from corpus."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search BM25 index for query."""
        pass
    
    @abstractmethod
    def search_documents(self, query: str, documents: List[Document], k: int = 10) -> List[Tuple[Document, float]]:
        """Search and return Document objects with scores."""
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """Save BM25 index to disk."""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load BM25 index from disk."""
        pass
    
    @abstractmethod
    def get_corpus_size(self) -> int:
        """Get the number of documents in the index."""
        pass
