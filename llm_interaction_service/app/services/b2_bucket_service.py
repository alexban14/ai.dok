import logging
import io
from typing import List
from b2sdk.v2 import *
from app.core.config import config
from app.interfaces.bucket_service_interface import BucketServiceInterface
import urllib3

logger = logging.getLogger(__name__)

class B2BucketService(BucketServiceInterface):
    def __init__(self):
        # Increase connection pool size for concurrent downloads
        # Default is 10, increase to match max_concurrent processing
        http_adapter = urllib3.PoolManager(
            maxsize=100,  # Maximum pool size
            block=False   # Don't block when pool is full
        )
        
        self.b2_api = B2Api(
            InMemoryAccountInfo(),
            cache=InMemoryCache(),
        )
        self.b2_api.authorize_account("production", config.b2_application_key_id, config.b2_application_key)
        self.bucket = self.b2_api.get_bucket_by_name(config.b2_bucket_name)
        
        logger.info(f"B2 connection pool initialized with maxsize=100")

    def list_files(self) -> List:
        return list(self.bucket.ls())

    def download_file_by_name(self, file_name: str) -> bytes:
        downloaded_file = self.bucket.download_file_by_name(file_name)
        return downloaded_file.response.content  # raw bytes
