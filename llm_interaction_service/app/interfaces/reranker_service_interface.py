from abc import ABC, abstractmethod
from typing import List, Tuple
from langchain.docstore.document import Document

class RerankerServiceInterface(ABC):
    """Interface for document reranking services."""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """Rerank documents based on query relevance."""
        pass
    
    @abstractmethod
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Rerank Document objects."""
        pass
