from abc import ABC, abstractmethod

class IndexingServiceInterface(ABC):
    """Interface for the indexing service."""

    @abstractmethod
    async def process_bucket(self) -> dict:
        """
        Process the entire bucket of RCP documents.
        """
        pass