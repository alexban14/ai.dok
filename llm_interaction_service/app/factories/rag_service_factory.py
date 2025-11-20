import logging
from typing import Optional
from app.services.rag_service import RagService
from app.interfaces.rag_service_interface import RagServiceInterface
from app.core.config import config
from app.core.constants import ChromaCollection

logger = logging.getLogger(__name__)

class RagServiceFactory:
    @staticmethod
    def create_rag_service(
        collection_name: Optional[str] = None,
        retrieval_strategy: Optional[str] = None
    ) -> RagServiceInterface:
        """Create RAG service with configurable collection and retrieval strategy."""
        collection = collection_name or ChromaCollection.RCP_DOCUMENTS_V2.value
        strategy = retrieval_strategy or config.retrieval_strategy
        
        logger.info(f"Creating RAG Service (collection={collection}, strategy={strategy})")
        return RagService(
            ollama_base_url=config.ollama_base_url,
            groq_api_key=config.groq_api_key,
            collection_name=collection,
            retrieval_strategy=strategy
        )
