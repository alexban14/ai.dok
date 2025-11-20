import logging
from typing import Optional
from app.services.indexing_service import IndexingService
from app.interfaces.indexing_service_interface import IndexingServiceInterface
from app.core.constants import ChromaCollection

logger = logging.getLogger(__name__)

class IndexingServiceFactory:
    @staticmethod
    def create_indexing_service(
        collection_name: Optional[str] = None,
        use_section_chunking: Optional[bool] = None
    ) -> IndexingServiceInterface:
        """Create indexing service with configurable collection and chunking strategy."""
        collection = collection_name or ChromaCollection.RCP_DOCUMENTS_V2.value
        
        logger.info(f"Creating Indexing Service (collection={collection})")
        return IndexingService(
            collection_name=collection,
            use_section_chunking=use_section_chunking
        )