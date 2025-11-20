import logging
from typing import Optional
from app.services.chroma_vector_store_service import ChromaVectorStoreService
from app.interfaces.vector_store_service_interface import VectorStoreServiceInterface
from app.core.constants import VectorStoreProvider

logger = logging.getLogger(__name__)

class VectorStoreServiceFactory:
    @staticmethod
    def create_vector_store_service(
        provider: str = "chroma",
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> VectorStoreServiceInterface:
        if provider == "chroma":
            logger.info(f"Creating Chroma Vector Store Service (collection: {collection_name})")
            return ChromaVectorStoreService(
                collection_name=collection_name,
                embedding_model=embedding_model
            )
        # Future providers can be added here
        raise ValueError(f"Unsupported vector store provider: {provider}")