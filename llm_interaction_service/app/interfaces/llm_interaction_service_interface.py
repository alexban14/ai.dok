from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any

class LlmInteractionServiceInterface(ABC):
    """Interface for LLM services (Ollama, Groq, etc.)"""

    @abstractmethod
    async def generate_completion(
            self,
            model: str,
            prompt: Dict[str, str],
            stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate text completion using the LLM service.

        Args:
            model (str): The model to use.
            prompt (str): The input prompt.
            stream (bool): If True, stream responses.

        Yields:
            dict: Response from the LLM service.
        """
        pass