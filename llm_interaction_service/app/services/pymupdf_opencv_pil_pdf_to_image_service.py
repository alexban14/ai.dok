import logging
import fitz
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image
import io
import asyncio
from typing import List, Tuple
from app.interfaces.pdf_to_image_service_interface import PDFToImageServiceInterface

logger = logging.getLogger(__name__)

class PyMuPDFOpenCvPilPDFToImageService(PDFToImageServiceInterface):
    """Service for converting PDF files to enhanced images."""

    async def convert_pdf_to_images(self, pdf_bytes: bytes, enhance: bool = True) -> List[bytes]:
        """
        Convert PDF to a list of image bytes.

        Args:
            pdf_bytes (bytes): The PDF file as bytes
            enhance (bool): Whether to enhance the image quality

        Returns:
            List[bytes]: List of bytes for each page image
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            logger.info(f"Converting PDF with {len(doc)} pages to images")

            image_bytes_list = []

            for page_num, page in enumerate(doc):
                logger.info(f"Processing page {page_num + 1}")
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)

                img_bytes = pix.tobytes("png")

                if enhance:
                    img_bytes = await self._enhance_image(img_bytes)

                image_bytes_list.append(img_bytes)

            logger.info(f"Successfully converted {len(image_bytes_list)} pages to images")
            return image_bytes_list

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error converting PDF to images: {str(e)}")

    async def process_pdf_file(self, file: UploadFile, enhance: bool = True) -> List[bytes]:
        """
        Process a PDF file uploaded through FastAPI.

        Args:
            file (UploadFile): The uploaded PDF file
            enhance (bool): Whether to enhance the image quality

        Returns:
            List[bytes]: List of bytes for each page image
        """
        try:
            pdf_bytes = await file.read()

            if not pdf_bytes:
                raise HTTPException(status_code=400, detail="Empty PDF file")

            return await self.convert_pdf_to_images(pdf_bytes, enhance)

        except Exception as e:
            logger.error(f"Failed to process PDF file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF file: {str(e)}")

    async def _enhance_image(self, img_bytes: bytes) -> bytes:
        """
        Enhance image quality for better OCR results.

        Args:
            img_bytes (bytes): Image as bytes

        Returns:
            bytes: Enhanced image as bytes
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Apply image enhancement techniques
            # 1. Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 3. Apply adaptive thresholding for better text extraction
            threshold = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # 4. Apply morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

            # 5. Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(opening)

            # Convert back to bytes
            _, buffer = cv2.imencode('.png', enhanced)
            enhanced_bytes = buffer.tobytes()

            return enhanced_bytes

        except Exception as e:
            logger.error(f"Failed to enhance image: {str(e)}")
            # Return original image if enhancement fails
            return img_bytes