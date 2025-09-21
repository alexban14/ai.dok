from abc import ABC, abstractmethod
from typing import List, Dict, Any
from fastapi import UploadFile

class OCRServiceInterface(ABC):
    """
    Abstract base class defining the interface for OCR services.
    This allows for easy swapping of OCR implementations.
    """

    @abstractmethod
    async def extract_text_from_image(self, image_bytes: bytes, lang: str = "eng") -> str:
        """
        Extract text from a single image.

        Args:
            image_bytes (bytes): The image as bytes
            lang (str): Language code for OCR (default: 'eng')

        Returns:
            str: Extracted text
        """
        pass

    @abstractmethod
    async def extract_text_from_multiple_images(self, image_bytes_list: List[bytes], lang: str = "eng") -> str:
        """
        Extract text from multiple images and combine the results.

        Args:
            image_bytes_list (List[bytes]): List of images as bytes
            lang (str): Language code for OCR (default: 'eng')

        Returns:
            str: Combined extracted text
        """
        pass

    @abstractmethod
    async def process_image_file(self, file: UploadFile, lang: str = "eng") -> str:
        """
        Process an image file uploaded through FastAPI.

        Args:
            file (UploadFile): The uploaded image file
            lang (str): Language code for OCR (default: 'eng')

        Returns:
            str: Extracted text
        """
        pass

    @abstractmethod
    async def extract_text_with_confidence(self, image_bytes: bytes, lang: str = "eng") -> Dict[str, Any]:
        """
        Extract text from an image with confidence scores.

        Args:
            image_bytes (bytes): The image as bytes
            lang (str): Language code for OCR (default: 'eng')

        Returns:
            Dict[str, Any]: Extracted text with confidence data
        """
        pass