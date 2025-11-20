import logging
from typing import Optional
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.interfaces.hybrid_retrieval_service_interface import HybridRetrievalServiceInterface
from app.interfaces.vector_store_service_interface import VectorStoreServiceInterface
from app.services.bm25_service import BM25Service
from app.services.reranker_service import RerankerService
from app.factories.vector_store_service_factory import VectorStoreServiceFactory
from app.factories.bm25_service_factory import BM25ServiceFactory
from app.factories.reranker_service_factory import RerankerServiceFactory
from app.core.config import config

logger = logging.getLogger(__name__)

class HybridRetrievalServiceFactory:
    """Factory for creating hybrid retrieval services."""
    
    @staticmethod
    def create_hybrid_retrieval_service(
        vector_store: Optional[VectorStoreServiceInterface] = None,
        bm25_service: Optional[BM25Service] = None,
        reranker_service: Optional[RerankerService] = None,
        collection_name: Optional[str] = None,
        alpha: Optional[float] = None,
        retrieval_top_k: Optional[int] = None,
        reranker_top_k: Optional[int] = None
    ) -> HybridRetrievalServiceInterface:
        """
        Create hybrid retrieval service with all dependencies.
        
        Args:
            vector_store: Vector store service (created if None)
            bm25_service: BM25 service (created if None)
            reranker_service: Reranker service (created if None)
            collection_name: Collection name for vector store and BM25
            alpha: Weight for hybrid fusion
            retrieval_top_k: Initial retrieval count
            reranker_top_k: Final result count
            
        Returns:
            HybridRetrievalService instance
        """
        logger.info("Creating Hybrid Retrieval Service")
        
        # Create dependencies if not provided
        if vector_store is None:
            logger.info("Creating vector store for hybrid retrieval")
            vector_store = VectorStoreServiceFactory.create_vector_store_service(
                collection_name=collection_name
            )
        
        if bm25_service is None:
            logger.info("Creating BM25 service for hybrid retrieval")
            bm25_service = BM25ServiceFactory.create_bm25_service(
                collection_name=collection_name
            )
        
        if reranker_service is None:
            logger.info("Creating reranker service for hybrid retrieval")
            reranker_service = RerankerServiceFactory.create_reranker_service()
        
        # Create hybrid service
        return HybridRetrievalService(
            vector_store=vector_store,
            bm25_service=bm25_service,
            reranker_service=reranker_service,
            alpha=alpha,
            retrieval_top_k=retrieval_top_k,
            reranker_top_k=reranker_top_k
        )
