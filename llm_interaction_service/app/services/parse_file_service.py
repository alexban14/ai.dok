import json
import re
import logging
import fitz
from fastapi import UploadFile, HTTPException
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.core.config import config
from app.core.constants import ProcessingType, AIService, OCRService, PDFToImageService
from app.factories.ocr_service_factory import OCRServiceFactory
from app.interfaces.ocr_service_interface import OCRServiceInterface
from app.factories.llm_interaction_service_factory import LlmInteractionServiceFactory
from app.factories.pdf_to_image_service_factory import PDFToImageServiceFactory
from app.interfaces.parse_file_service_interface import ParseFileServiceInterface

logger = logging.getLogger(__name__)

# Constants
PAGE_LOG_INTERVAL = 10

class ParseFileService(ParseFileServiceInterface):
    def __init__(self, ollama_base_url: str = "http://llm_host_service:11434", groq_api_key: str = None):
        """
        Initialize the ParseFileService class

        Args:
            ollama_base_url (str): Base URL for Ollama service.
            groq_api_key (str, optional): API key for Groq service.
        """
        self._llm_service = None
        self.pdf_to_image_service = PDFToImageServiceFactory.create_pdf_to_image_service(
            service_name=PDFToImageService.PYMUPDF_OPENCV_PILLOW
        )
        self._ocr_service = None
        self.ollama_base_url = ollama_base_url
        self.groq_api_key = groq_api_key

    async def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from a PDF using PyMuPDF."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            # TODO: DELETE
            logger.info(f"ParseFileService - opened stream of bytes")

            pages_text = []
            # Process page by page to manage memory
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                pages_text.append(page_text)
                if page_num % PAGE_LOG_INTERVAL == 0:
                    logger.debug(f"Processed {page_num + 1} pages")
            
            extracted_text = "\n".join(pages_text).strip()
            doc.close()

            # TODO: DELETE
            logger.info(f"ParseFileService - Length of extracted text: {len(extracted_text)}")

            if len(extracted_text) < 50:
                return "__SCANNED_DOCUMENT__"

            return extracted_text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise HTTPException(status_code=500, detail="Error extracting text from PDF.")

    async def process_with_ocr(self, pdf_bytes: bytes) -> str:
        """Process a scanned PDF using OCR."""
        logger.info("Converting PDF to images for OCR processing")
        images = await self.pdf_to_image_service.convert_pdf_to_images(pdf_bytes, enhance=True)

        if not images:
            raise HTTPException(status_code=500, detail="Failed to convert PDF to images")

        logger.info(f"Starting OCR processing on {len(images)} images")
        extracted_text = await self._ocr_service.extract_text_from_multiple_images(images)

        if not extracted_text:
            logger.warning("OCR processing completed but no text was extracted")
            raise HTTPException(
                status_code=400,
                detail="The document appears to be blank or OCR could not extract text"
            )

        logger.info(f"OCR processing completed. Extracted {len(extracted_text)} characters.")
        return extracted_text

    def _create_parse_prompt(self, extracted_text: str) -> Dict[str, str]:
        context = f"""
            RCP text data:
            {extracted_text}
        """
        user_input = """
            Extract and structure data from the provided RCP text. The goal is to identify key information for a medical professional. 
            The returned data should be of type JSON blob.
            The JSON must have the following structure:
            {{
                "drug_name": "<drug_name>",
                "interactions": "<Text from section 4.5...>",
                "adverse_reactions": "<Text from section 4.8...>",
                "pregnancy_lactation": "<Text from section 4.6...>"
            }}
        """

        return {"system": context, "user": user_input}

    def _create_custom_prompt(self, extracted_text: str, user_prompt: str) -> Dict[str, str]:
        """Create a custom prompt based on user input."""
        system = f"""
            System:

            You are an AI medical assistant. Your role is to provide concise and accurate information to doctors based on official medical documents (RCPs).
            Analyze the provided text from the RCP document to answer the user's question.
            The returned data should be of type JSON blob. The JSON must have a single key called "response", under that key,
            the value should be a string representing html format so that it can be displayed in a web app using "innerHTML".

            File Context:
            {extracted_text}
        """
        user = f"""
            User Prompt: {user_prompt}
        """

        return {"system": system, "user": user}

    async def _process_with_rag(self, extracted_text: str, prompt: str, model: str) -> str:
        """Process file with RAG approach (for Groq service)."""
        try:
            retrieved_docs = self.vector_store_service.similarity_search(prompt, k=5)
            retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

            combined_context = f"""Uploaded file content:\n{extracted_text}\n\nRetrieved additional context from knowledge base:\n{retrieved_text}"""

            prompt_for_llm = self._create_custom_prompt(combined_context, prompt)

            result = ""
            async for chunk in self._llm_service.generate_completion(
                    model=model,
                    prompt=prompt_for_llm,
                    stream=False
            ):
                result += chunk["response"]

            cleaned_text = re.sub(r"^```json\n|\n```$", "", result).strip()

            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"ParseFileService - Failed to parse JSON response: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing with RAG: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing with RAG: {str(e)}")

    async def process(
            self,
            model: str,
            file: UploadFile,
            processing_type: str,
            prompt: str = None,
            ai_service: str = AIService.GROQ_CLOUD,
            ocr_technology: str = OCRService.PADDLE
    ) -> str:
        self._ocr_service = OCRServiceFactory.create_ocr_service(
            service_name=ocr_technology
        )
        self._llm_service = LlmInteractionServiceFactory.create_llm_interaction_service(
            ai_service,
            self.ollama_base_url,
            self.groq_api_key
        )

        pdf_bytes = await file.read()
        extracted_text = await self.extract_text_from_pdf(pdf_bytes)

        if extracted_text == "__SCANNED_DOCUMENT__":
            extracted_text = await self.process_with_ocr(pdf_bytes)

        if processing_type == ProcessingType.PARSE:
            parse_prompt = self._create_parse_prompt(extracted_text)

            result = ""
            async for chunk in self._llm_service.generate_completion(model=model, prompt=parse_prompt, stream=False):
                result += chunk["response"]

            cleaned_text = re.sub(r"^```json\n|\n```$", "", result).strip()

            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse JSON response: {str(e)}")

        elif processing_type == ProcessingType.PROMPT:
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required for 'prompt' type.")

            if ai_service == AIService.GROQ_CLOUD:
                return await self._process_with_rag(extracted_text, prompt, model)

            custom_prompt = self._create_custom_prompt(extracted_text, prompt)

            results = ""
            async for chunk in self._llm_service.generate_completion(model=model, prompt=custom_prompt, stream=False):
                results += chunk["response"]

            cleaned_text = re.sub(r"^```json\n|\n```$", "", results).strip()

            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse JSON response: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Invalid processing type. Use 'parse' or 'prompt'.")