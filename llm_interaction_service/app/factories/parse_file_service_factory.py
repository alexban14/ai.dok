import logging
from app.services.parse_file_service import ParseFileService
from app.interfaces.parse_file_service_interface import ParseFileServiceInterface
from app.core.config import config

logger = logging.getLogger(__name__)

class ParseFileServiceFactory:
    @staticmethod
    def create_parse_file_service() -> ParseFileServiceInterface:
        logger.info("Creating Parse File Service")
        return ParseFileService(
            ollama_base_url=config.ollama_base_url,
            groq_api_key=config.groq_api_key
        )