import logging
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image
import io
from typing import List, Dict, Any, Optional
from paddleocr import PaddleOCR
from app.interfaces.ocr_service_interface import OCRServiceInterface

logger = logging.getLogger(__name__)

class PaddleOCRService(OCRServiceInterface):
    """Service for extracting text from images using PaddleOCR."""

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        """
        Initialize the PaddleOCR service.

        Args:
            lang (str): Language code for OCR (default: 'en')
            use_gpu (bool): Whether to use GPU if available (default: False)
        """
        try:
            self.ocr_engine = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                use_gpu=use_gpu,
                show_log=False,
                enable_mkldnn=True  # Enable Intel MKL-DNN acceleration
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initialize PaddleOCR")

    async def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for better OCR results."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            return processed
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            # Return original image if preprocessing fails
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    async def extract_text_from_image(self, image_bytes: bytes, lang: str = "en") -> str:
        """
        Extract text from an image using PaddleOCR.

        Args:
            image_bytes (bytes): The image as bytes
            lang (str): Language code for OCR (default: 'en')

        Returns:
            str: Extracted text
        """
        try:
            # Preprocess image
            img = await self._preprocess_image(image_bytes)

            # Perform OCR
            result = self.ocr_engine.ocr(img, cls=True)

            # Extract and combine text
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and line[1]:
                        text_lines.append(line[1][0])

            return "\n".join(text_lines).strip()

        except Exception as e:
            logger.error(f"Failed to extract text from image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting text with PaddleOCR: {str(e)}")

    async def extract_text_from_multiple_images(self, image_bytes_list: List[bytes], lang: str = "en") -> str:
        """
        Extract text from multiple images and combine the results.

        Args:
            image_bytes_list (List[bytes]): List of images as bytes
            lang (str): Language code for OCR (default: 'en')

        Returns:
            str: Combined extracted text
        """
        if not image_bytes_list:
            return ""

        texts = []

        for i, img_bytes in enumerate(image_bytes_list):
            logger.info(f"Processing image {i+1} of {len(image_bytes_list)} with PaddleOCR")
            text = await self.extract_text_from_image(img_bytes, lang)
            texts.append(text)

        return "\n\n".join(texts)

    async def process_image_file(self, file: UploadFile, lang: str = "en") -> str:
        """
        Process an image file uploaded through FastAPI.

        Args:
            file (UploadFile): The uploaded image file
            lang (str): Language code for OCR (default: 'en')

        Returns:
            str: Extracted text
        """
        try:
            image_bytes = await file.read()

            if not image_bytes:
                raise HTTPException(status_code=400, detail="Empty image file")

            return await self.extract_text_from_image(image_bytes, lang)

        except Exception as e:
            logger.error(f"Failed to process image file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing image file: {str(e)}")

    async def extract_text_with_confidence(self, image_bytes: bytes, lang: str = "en") -> Dict[str, Any]:
        """
        Extract text from an image and include confidence scores.

        Args:
            image_bytes (bytes): The image as bytes
            lang (str): Language code for OCR (default: 'en')

        Returns:
            Dict[str, Any]: Extracted text with confidence data
        """
        try:
            # Preprocess image
            img = await self._preprocess_image(image_bytes)

            # Perform OCR
            result = self.ocr_engine.ocr(img, cls=True)

            # Process results
            text_blocks = []
            confidence_sum = 0
            confidence_count = 0

            if result and result[0]:
                for i, line in enumerate(result[0]):
                    if line and line[1]:
                        text = line[1][0]
                        confidence = line[1][1]

                        text_blocks.append({
                            'text': text,
                            'confidence': confidence,
                            'block_num': i,
                            'line_num': i  # PaddleOCR doesn't provide line numbers directly
                        })

                        confidence_sum += confidence
                        confidence_count += 1

            # Calculate average confidence
            avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0

            # Construct full text
            full_text = '\n'.join([block['text'] for block in text_blocks])

            return {
                'text': full_text,
                'avg_confidence': avg_confidence,
                'blocks': text_blocks
            }

        except Exception as e:
            logger.error(f"Failed to extract text with confidence: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting text with confidence: {str(e)}")
