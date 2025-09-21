from abc import ABC, abstractmethod
from typing import List
from fastapi import UploadFile

class PDFToImageServiceInterface(ABC):
    """
    Abstract base class defining the interface for PDF to Image conversion services.
    This allows for easy swapping of PDF conversion implementations.
    """

    @abstractmethod
    async def convert_pdf_to_images(self, pdf_bytes: bytes, enhance: bool = True) -> List[bytes]:
        """
        Convert PDF to a list of image bytes.

        Args:
            pdf_bytes (bytes): The PDF file as bytes
            enhance (bool): Whether to enhance the image quality

        Returns:
            List[bytes]: List of bytes for each page image
        """
        pass

    @abstractmethod
    async def process_pdf_file(self, file: UploadFile, enhance: bool = True) -> List[bytes]:
        """
        Process a PDF file uploaded through FastAPI.

        Args:
            file (UploadFile): The uploaded PDF file
            enhance (bool): Whether to enhance the image quality

        Returns:
            List[bytes]: List of bytes for each page image
        """
        pass