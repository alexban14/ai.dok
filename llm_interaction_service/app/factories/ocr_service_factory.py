import logging
from typing import Optional

from app.interfaces.ocr_service_interface import OCRServiceInterface
from app.services.paddle_ocr_service import PaddleOCRService
from app.core.constants import OCRService

logger = logging.getLogger(__name__)

class OCRServiceFactory:
    """
    Factory class for creating OCR services based on configuration.
    Allows easy addition of new OCR service implementations.
    """

    @staticmethod
    def create_ocr_service(
        service_name: str,
        lang: str = "en",
        use_gpu: bool = False
    ) -> OCRServiceInterface:
        """
        Create an OCR service based on the configuration.

        Args:
            service_name (str): Name of the OCR service to create

        Returns:
            OCRServiceInterface: An instance of the specified OCR service
        """
        service_name = service_name.lower()

        if service_name == OCRService.PADDLE:
            logger.info("Creating Paddle OCR Service")
            return PaddleOCRService(lang, use_gpu)

        raise ValueError(f"Unsupported OCR service: {service_name}")