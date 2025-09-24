import logging
from app.services.b2_bucket_service import B2BucketService
from app.interfaces.bucket_service_interface import BucketServiceInterface
from app.core.constants import BucketProvider

logger = logging.getLogger(__name__)

class BucketServiceFactory:
    @staticmethod
    def create_bucket_service(provider: str = BucketProvider.B2) -> BucketServiceInterface:
        if provider == BucketProvider.B2:
            logger.info("Creating B2 Bucket Service")
            return B2BucketService()
        # Future providers can be added here
        raise ValueError(f"Unsupported bucket provider: {provider}")