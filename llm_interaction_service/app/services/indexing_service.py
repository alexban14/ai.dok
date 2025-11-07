import logging
import asyncio
from b2sdk.v2 import *
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import config
from app.factories.parse_file_service_factory import ParseFileServiceFactory
from app.factories.vector_store_service_factory import VectorStoreServiceFactory
from app.interfaces.indexing_service_interface import IndexingServiceInterface
from app.factories.bucket_service_factory import BucketServiceFactory

logger = logging.getLogger(__name__)

class IndexingService(IndexingServiceInterface):
    def __init__(self):
        # Create services using factories
        self.bucket_service = BucketServiceFactory.create_bucket_service()
        self.parse_file_service = ParseFileServiceFactory.create_parse_file_service()
        self.vector_store_service = VectorStoreServiceFactory.create_vector_store_service()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    async def _process_file(self, file_info, total_files, current_index):
        logger.info(f"Processing file {current_index}/{total_files}: {file_info.file_name}")
        try:
            # 1. Download PDF from bucket (in thread)
            logger.debug(f"Downloading {file_info.file_name}...")
            pdf_bytes = await asyncio.to_thread(self.bucket_service.download_file_by_name, file_info.file_name)
            logger.debug(f"Downloaded {len(pdf_bytes)} bytes.")

            # 2. Extract text from PDF (with OCR fallback)
            logger.debug(f"Extracting text from {file_info.file_name}...")
            extracted_text = await self.parse_file_service.extract_text_from_pdf(pdf_bytes)

            if extracted_text == "__SCANNED_DOCUMENT__":
                logger.info(f"{file_info.file_name} is a scanned document, using OCR.")
                extracted_text = await self.parse_file_service.process_with_ocr(pdf_bytes)

            logger.debug(f"Extracted {len(extracted_text)} characters.")

            # 3. Chunk the extracted text (in thread)
            logger.debug(f"Chunking text for {file_info.file_name}...")
            chunks = await asyncio.to_thread(self.text_splitter.split_text, extracted_text)
            logger.debug(f"Created {len(chunks)} chunks.")

            # 4. Vectorize and store the chunks (in thread)
            logger.debug(f"Vectorizing and storing chunks for {file_info.file_name}...")
            ids = [f"{file_info.file_name}-{i}" for i, _ in enumerate(chunks)]
            await asyncio.to_thread(
                self.vector_store_service.add_texts,
                texts=chunks,
                metadatas=[{"source": file_info.file_name}] * len(chunks),
                ids=ids
            )

            logger.info(f"Successfully processed and stored {file_info.file_name}.")
            return {"status": "success", "file_name": file_info.file_name}
        except Exception as e:
            error_message = f"Failed to process file {file_info.file_name}: {e}"
            logger.error(error_message)
            return {"status": "failed", "file_name": file_info.file_name, "error": str(e)}

    async def process_bucket(self) -> dict:
        logger.info("Starting bucket processing...")

        # List all files in the bucket
        all_files = [file_info for file_info, _ in self.bucket_service.list_files() if file_info.file_name.lower().endswith('.pdf')]
        total_files = len(all_files)
        logger.info(f"Found {total_files} PDF files in the bucket.")

        tasks = []
        for i, file_info in enumerate(all_files):
            tasks.append(self._process_file(file_info, total_files, i + 1))

        results = await asyncio.gather(*tasks)

        processed_files_count = sum(1 for r in results if r['status'] == 'success')
        failed_files = [r for r in results if r['status'] == 'failed']

        logger.info("Bucket processing finished.")
        return {
            "message": "Bucket processing completed.",
            "total_pdf_files_in_bucket": total_files,
            "processed_pdf_files": processed_files_count,
            "failed_files": failed_files
        }
