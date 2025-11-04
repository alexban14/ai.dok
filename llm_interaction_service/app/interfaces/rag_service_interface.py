from abc import ABC, abstractmethod
from typing import Dict

class RagServiceInterface(ABC):
    """Interface for the RAG service."""

    @abstractmethod
    async def query(self, model: str, prompt: str, ai_service: str, collection_name: str) -> Dict:
        """
        Query the RAG pipeline.
        """
        pass
