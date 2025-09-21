import logging

from app.interfaces.llm_interaction_service_interface import LlmInteractionServiceInterface
from app.services.ollama_service import OllamaService
from app.services.groq_service import GroqService

logger = logging.getLogger(__name__)

class LlmInteractionServiceFactory:
    """
    Factory class for creating LLM interaction services based on configuration.
    Allows easy addition of new LLM interaction service implementations.
    """

    @staticmethod
    def create_llm_interaction_service(ai_service: str, ollama_base_url: str, groq_api_key: str) -> LlmInteractionServiceInterface:
        if ai_service == "ollama_local":
            return OllamaService(base_url=ollama_base_url)
        elif ai_service == "groq_cloud":
            return GroqService(api_key=groq_api_key)
        # Add other service conditions here in the future

        raise ValueError(f"Unsupported AI service: {ai_service}")