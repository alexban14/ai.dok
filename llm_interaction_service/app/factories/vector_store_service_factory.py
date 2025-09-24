import logging
from app.services.chroma_vector_store_service import ChromaVectorStoreService
from app.interfaces.vector_store_service_interface import VectorStoreServiceInterface
from app.core.constants import VectorStoreProvider

logger = logging.getLogger(__name__)

class VectorStoreServiceFactory:
    @staticmethod
    def create_vector_store_service(provider: str = "chroma") -> VectorStoreServiceInterface:
        if provider == "chroma":
            logger.info("Creating Chroma Vector Store Service")
            return ChromaVectorStoreService()
        # Future providers can be added here
        raise ValueError(f"Unsupported vector store provider: {provider}")