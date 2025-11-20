import logging
from typing import Optional
from pathlib import Path
from app.services.bm25_service import BM25Service
from app.interfaces.bm25_service_interface import BM25ServiceInterface
from app.core.config import config
from app.core.constants import ChromaCollection

logger = logging.getLogger(__name__)

class BM25ServiceFactory:
    """Factory for creating BM25 services."""
    
    @staticmethod
    def create_bm25_service(
        collection_name: Optional[str] = None,
        index_path: Optional[str] = None
    ) -> BM25ServiceInterface:
        """
        Create BM25 service instance.
        
        Args:
            collection_name: ChromaDB collection name (for index path naming)
            index_path: Custom path to BM25 index file
            
        Returns:
            BM25Service instance
        """
        # Determine index path
        if index_path is None:
            collection = collection_name or ChromaCollection.RCP_DOCUMENTS_V2.value
            index_path = f"data/bm25_index_{collection}.pkl"
        
        logger.info(f"Creating BM25 Service with index: {index_path}")
        
        # Try to load existing index
        if Path(index_path).exists():
            logger.info(f"Loading existing BM25 index from {index_path}")
            return BM25Service(index_path=index_path)
        else:
            logger.warning(f"BM25 index not found at {index_path}, creating empty service")
            return BM25Service()
