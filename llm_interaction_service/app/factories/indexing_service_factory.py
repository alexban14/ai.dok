import logging
from app.services.indexing_service import IndexingService
from app.interfaces.indexing_service_interface import IndexingServiceInterface

logger = logging.getLogger(__name__)

class IndexingServiceFactory:
    @staticmethod
    def create_indexing_service() -> IndexingServiceInterface:
        logger.info("Creating Indexing Service")
        return IndexingService()