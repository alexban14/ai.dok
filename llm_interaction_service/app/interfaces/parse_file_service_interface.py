from abc import ABC, abstractmethod
from fastapi import UploadFile
from typing import Dict, Any

class ParseFileServiceInterface(ABC):
    """Interface for processing file parsing requests."""

    @abstractmethod
    async def process(
            self,
            model: str,
            file: UploadFile,
            processing_type: str,
            prompt: str = None,
            ai_service: str = "ollama_local"
    ) -> str:
        """
        Process file using the specified AI service.

        Args:
            model (str): The model to use.
            file (UploadFile): The invoice file to process.
            processing_type (str): Type of processing ("parse" or "prompt").
            prompt (str, optional): Custom prompt for LLM. Required for "prompt" type.
            ai_service (str): AI service to use ("ollama_local" or "groq_cloud").

        Returns:
            Dict[str, Any]: The processed file data.
        """
        pass