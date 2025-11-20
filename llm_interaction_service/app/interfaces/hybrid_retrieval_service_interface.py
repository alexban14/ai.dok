from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from langchain.docstore.document import Document

class HybridRetrievalServiceInterface(ABC):
    """Interface for hybrid retrieval services."""
    
    @abstractmethod
    def retrieve(self, query: str, strategy: str = "hybrid", k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Retrieve documents using specified strategy."""
        pass
    
    @abstractmethod
    def retrieve_with_metadata(self, query: str, strategy: str = "hybrid", k: Optional[int] = None) -> Dict[str, any]:
        """Retrieve documents with detailed metadata about retrieval process."""
        pass
