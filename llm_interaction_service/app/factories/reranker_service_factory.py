import logging
from app.services.reranker_service import RerankerService
from app.interfaces.reranker_service_interface import RerankerServiceInterface
from app.core.config import config

logger = logging.getLogger(__name__)

class RerankerServiceFactory:
    """Factory for creating reranker services."""
    
    @staticmethod
    def create_reranker_service(model_name: str = None) -> RerankerServiceInterface:
        """
        Create reranker service instance.
        
        Args:
            model_name: HuggingFace model name (default from config)
            
        Returns:
            RerankerService instance
        """
        model_name = model_name or config.reranker_model
        logger.info(f"Creating Reranker Service with model: {model_name}")
        
        return RerankerService(model_name=model_name)
