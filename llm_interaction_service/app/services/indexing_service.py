import logging
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

    async def process_bucket(self) -> dict:
        logger.info("Starting bucket processing...")
        processed_files_count = 0
        failed_files = []

        # List all files in the bucket
        all_files = self.bucket_service.list_files()
        total_files = len(all_files)
        logger.info(f"Found {total_files} files in the bucket.")

        for i, (file_info, _) in enumerate(all_files):
            if not file_info.file_name.lower().endswith('.pdf'):
                continue

            logger.info(f"Processing file {i+1}/{total_files}: {file_info.file_name}")

            try:
                # 1. Download PDF from bucket
                logger.debug(f"Downloading {file_info.file_name}...")
                pdf_bytes = self.bucket_service.download_file_by_name(file_info.file_name)
                logger.debug(f"Type of pdf_bytes: {type(pdf_bytes)}")
                # pdf_bytes = file_download.read()
                logger.debug(f"Downloaded {len(pdf_bytes)} bytes.")

                # 2. Extract text from PDF (with OCR fallback)
                logger.debug(f"Extracting text from {file_info.file_name}...")
                extracted_text = await self.parse_file_service.extract_text_from_pdf(pdf_bytes)

                if extracted_text == "__SCANNED_DOCUMENT__":
                    logger.info(f"{file_info.file_name} is a scanned document, using OCR.")
                    extracted_text = await self.parse_file_service.process_with_ocr(pdf_bytes)

                logger.debug(f"Extracted {len(extracted_text)} characters.")

                # 3. Chunk the extracted text
                logger.debug(f"Chunking text for {file_info.file_name}...")
                chunks = self.text_splitter.split_text(extracted_text)
                logger.debug(f"Created {len(chunks)} chunks.")

                # 4. Vectorize and store the chunks
                logger.debug(f"Vectorizing and storing chunks for {file_info.file_name}...")
                self.vector_store_service.add_texts(
                    texts=chunks,
                    metadatas=[{"source": file_info.file_name}] * len(chunks)
                )

                logger.info(f"Successfully processed and stored {file_info.file_name}.")
                processed_files_count += 1

            except Exception as e:
                error_message = f"Failed to process file {file_info.file_name}: {e}"
                logger.error(error_message)
                failed_files.append({"file": file_info.file_name, "error": str(e)})

        logger.info("Bucket processing finished.")
        return {
            "message": "Bucket processing completed.",
            "total_files_in_bucket": total_files,
            "processed_pdf_files": processed_files_count,
            "failed_files": failed_files
        }