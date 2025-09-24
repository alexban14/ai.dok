from abc import ABC, abstractmethod
from typing import List

class BucketServiceInterface(ABC):
    """Interface for bucket services."""

    @abstractmethod
    def list_files(self) -> List:
        """List all files in the bucket."""
        pass

    @abstractmethod
    def download_file_by_name(self, file_name: str) -> bytes:
        """Download a file from the bucket by its name."""
        pass