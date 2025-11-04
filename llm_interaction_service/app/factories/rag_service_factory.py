import logging
from app.services.rag_service import RagService
from app.interfaces.rag_service_interface import RagServiceInterface
from app.core.config import config

logger = logging.getLogger(__name__)

class RagServiceFactory:
    @staticmethod
    def create_rag_service() -> RagServiceInterface:
        logger.info("Creating RAG Service")
        return RagService(
            ollama_base_url=config.ollama_base_url,
            groq_api_key=config.groq_api_key
        )
