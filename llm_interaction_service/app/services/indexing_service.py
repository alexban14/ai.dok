import logging
import asyncio
import gc
from pathlib import Path
from b2sdk.v2 import *
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import config
from app.core.constants import ChromaCollection
from app.factories.parse_file_service_factory import ParseFileServiceFactory
from app.factories.vector_store_service_factory import VectorStoreServiceFactory
from app.interfaces.indexing_service_interface import IndexingServiceInterface
from app.factories.bucket_service_factory import BucketServiceFactory
from app.services.rcp_section_parser_service import RCPSectionParserService
from app.services.bm25_service import BM25Service

logger = logging.getLogger(__name__)

class IndexingService(IndexingServiceInterface):
    def __init__(self, collection_name: str = None, use_section_chunking: bool = None):
        """
        Initialize indexing service.
        
        Args:
            collection_name: ChromaDB collection name (default: rcp_documents_v2)
            use_section_chunking: Use RCP section-aware chunking (default: from config)
        """
        # Create services using factories
        self.bucket_service = BucketServiceFactory.create_bucket_service()
        self.parse_file_service = ParseFileServiceFactory.create_parse_file_service()
        
        # Use new collection for BGE embeddings
        self.collection_name = collection_name or ChromaCollection.RCP_DOCUMENTS_V2.value
        self.vector_store_service = VectorStoreServiceFactory.create_vector_store_service(
            collection_name=self.collection_name
        )

        # RCP section parser
        self.section_parser = RCPSectionParserService()
        
        # Configuration
        self.use_section_chunking = (
            use_section_chunking if use_section_chunking is not None 
            else config.chunk_by_section
        )
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        
        # Initialize text splitter (fallback for non-section chunking)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # BM25 index for keyword search
        self.bm25_service = None
        self.bm25_index_path = f"data/bm25_index_{self.collection_name}.pkl"
        
        logger.info(
            f"Indexing service initialized: collection={self.collection_name}, "
            f"section_chunking={self.use_section_chunking}, "
            f"chunk_size={self.chunk_size}"
        )

    async def _process_file(self, file_info, total_files, current_index):
        """Process a single file with optimized async operations and memory management."""
        # Force garbage collection before processing to free memory
        gc.collect()
        
        # Only log every 10th file to reduce I/O overhead
        if current_index % 10 == 0 or current_index == 1:
            logger.info(f"Processing file {current_index}/{total_files}: {file_info.file_name}")
        
        try:
            # 1. Download PDF from bucket (in thread)
            pdf_bytes = await asyncio.to_thread(self.bucket_service.download_file_by_name, file_info.file_name)

            # 2. Extract text from PDF (with OCR fallback)
            extracted_text = await self.parse_file_service.extract_text_from_pdf(pdf_bytes)

            if extracted_text == "__SCANNED_DOCUMENT__":
                extracted_text = await self.parse_file_service.process_with_ocr(pdf_bytes)

            # 3. Chunk the extracted text
            
            if self.use_section_chunking:
                # Parse RCP sections and create section-aware chunks
                chunks_data = await self._chunk_by_sections(extracted_text, file_info.file_name)
            else:
                # Use standard recursive text splitter (fast, no thread needed)
                chunks_text = self.text_splitter.split_text(extracted_text)
                chunks_data = [
                    {
                        'text': chunk,
                        'metadata': {
                            'source': file_info.file_name,
                            'chunk_index': i,
                            'chunking_method': 'recursive'
                        }
                    }
                    for i, chunk in enumerate(chunks_text)
                ]

            # 4. Vectorize and store chunks (critical path - keep in thread)
            
            texts = [chunk['text'] for chunk in chunks_data]
            metadatas = [chunk['metadata'] for chunk in chunks_data]
            ids = [f"{file_info.file_name}-{i}" for i in range(len(chunks_data))]
            
            # Use thread pool for vectorization to avoid blocking
            await asyncio.to_thread(
                self.vector_store_service.add_texts,
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )

            # Only log success for every 10th file
            if current_index % 10 == 0:
                logger.info(f"✓ Processed {current_index}/{total_files} files")
            
            result = {
                "status": "success",
                "file_name": file_info.file_name,
                "chunks": len(chunks_data),
                "texts": texts  # Return for BM25 indexing
            }
            
            # Cleanup large objects (GC every 20 files, not every file)
            del pdf_bytes, extracted_text, chunks_data, texts, metadatas, ids
            if current_index % 20 == 0:
                gc.collect()
            
            return result
        except Exception as e:
            error_message = f"Failed to process file {file_info.file_name}: {e}"
            logger.error(error_message, exc_info=True)
            return {"status": "failed", "file_name": file_info.file_name, "error": str(e)}
    
    async def _chunk_by_sections(self, text: str, source_file: str) -> list:
        """
        Chunk text by RCP sections.
        
        Args:
            text: Full RCP document text
            source_file: Source file name
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Parse sections (fast operation, no thread needed)
        sections = self.section_parser.parse_sections(text)
        
        if not sections:
            # Fallback chunking (also fast)
            chunks_text = self.text_splitter.split_text(text)
            return [
                {
                    'text': chunk,
                    'metadata': {
                        'source': source_file,
                        'chunk_index': i,
                        'chunking_method': 'fallback'
                    }
                }
                for i, chunk in enumerate(chunks_text)
            ]
        
        # Create chunks from sections (fast operation)
        chunks_data = self.section_parser.create_chunks_from_sections(
            sections,
            max_chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            preserve_sections=True
        )
        
        # Add source to metadata
        for chunk in chunks_data:
            chunk['metadata']['source'] = source_file
            chunk['metadata']['chunking_method'] = 'section_aware'
        
        logger.debug(f"Created {len(chunks_data)} section-aware chunks from {len(sections)} sections")
        return chunks_data

    async def process_bucket(self, batch_size: int = 10, max_concurrent: int = 5) -> dict:
        """
        Process bucket with batching and concurrency control for faster indexing.
        
        Args:
            batch_size: Number of files to process per batch for BM25 indexing
            max_concurrent: Maximum concurrent file processing tasks
        """
        logger.info(f"Starting bucket processing (concurrent={max_concurrent}, batch_size={batch_size})...")

        # List all files in the bucket
        all_files = [file_info for file_info, _ in self.bucket_service.list_files() if file_info.file_name.lower().endswith('.pdf')]
        total_files = len(all_files)
        logger.info(f"Found {total_files} PDF files in the bucket.")

        # Process files with concurrency limit using semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_info, index):
            async with semaphore:
                return await self._process_file(file_info, total_files, index)
        
        tasks = [
            process_with_semaphore(file_info, i + 1)
            for i, file_info in enumerate(all_files)
        ]
        
        # Process with progress updates
        results = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                logger.info(f"Progress: {completed}/{total_files} files processed ({completed/total_files*100:.1f}%)")
        
        logger.info(f"All {total_files} files processed!")

        
        processed_files_count = sum(1 for r in results if r['status'] == 'success')
        failed_files = [r for r in results if r['status'] == 'failed']
        
        # Build BM25 index from all processed texts in batches
        logger.info("Building BM25 index from all processed texts...")
        all_texts = []
        for result in results:
            if result['status'] == 'success' and 'texts' in result:
                all_texts.extend(result['texts'])
        
        if all_texts:
            await self._build_bm25_index(all_texts)
            logger.info(f"BM25 index built with {len(all_texts)} documents")
        else:
            logger.warning("No texts available for BM25 index")

        logger.info("Bucket processing finished.")
        return {
            "message": "Bucket processing completed.",
            "collection_name": self.collection_name,
            "total_pdf_files_in_bucket": total_files,
            "processed_pdf_files": processed_files_count,
            "total_chunks": sum(r.get('chunks', 0) for r in results if r['status'] == 'success'),
            "bm25_corpus_size": len(all_texts),
            "failed_files": failed_files,
            "processing_stats": {
                "max_concurrent": max_concurrent,
                "batch_size": batch_size
            }
        }
    
    async def _build_bm25_index(self, texts: list):
        """
        Build and save BM25 index from texts.
        
        Args:
            texts: List of text chunks
        """
        try:
            # Build BM25 index in thread (CPU-intensive)
            self.bm25_service = BM25Service(corpus=texts)
            
            # Save index to disk
            await asyncio.to_thread(
                self.bm25_service.save_index,
                self.bm25_index_path
            )
            
            logger.info(f"BM25 index saved to {self.bm25_index_path}")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
    
    def get_bm25_service(self) -> BM25Service:
        """
        Get BM25 service, loading from disk if needed.
        
        Returns:
            BM25Service instance
        """
        if self.bm25_service is None:
            if Path(self.bm25_index_path).exists():
                logger.info(f"Loading BM25 index from {self.bm25_index_path}")
                self.bm25_service = BM25Service(index_path=self.bm25_index_path)
            else:
                logger.warning("BM25 index not found, please run indexing first")
        
        return self.bm25_service
    
    def process_bucket_sync(self, batch_size: int = 10, max_concurrent: int = 5, job_id_param: str = None) -> dict:
        """
        Synchronous wrapper for process_bucket to run in multiprocessing.
        This method is called from a separate process via BackgroundJobManager.
        
        Args:
            batch_size: Number of files to process per batch for BM25 indexing
            max_concurrent: Maximum concurrent file processing tasks
            job_id_param: Job ID for progress updates
            
        Returns:
            Dictionary with processing results
        """
        import asyncio
        from app.services.background_job_manager import job_manager
        
        logger.info(f"[Job {job_id_param}] Starting sync processing wrapper")
        
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run async process_bucket in this loop
            result = loop.run_until_complete(
                self._process_bucket_with_progress(batch_size, max_concurrent, job_id_param)
            )
            return result
        finally:
            loop.close()
    
    async def _process_bucket_with_progress(self, batch_size: int, max_concurrent: int, job_id: str) -> dict:
        """
        Process bucket with progress updates to job manager.
        """
        from app.services.background_job_manager import job_manager
        
        logger.info(f"[Job {job_id}] Starting bucket processing...")

        # List all files
        all_files = [file_info for file_info, _ in self.bucket_service.list_files() if file_info.file_name.lower().endswith('.pdf')]
        total_files = len(all_files)
        logger.info(f"[Job {job_id}] Found {total_files} PDF files")
        
        # Update job with total count
        if job_id:
            job_manager.update_progress(job_id, 0, total_files)
        
        # Process files with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file_info, index):
            async with semaphore:
                result = await self._process_file(file_info, total_files, index)
                # Update progress after each file
                if job_id:
                    job_manager.update_progress(
                        job_id, 
                        index, 
                        total_files, 
                        current_file=file_info.file_name
                    )
                return result
        
        tasks = [
            process_with_semaphore(file_info, i + 1)
            for i, file_info in enumerate(all_files)
        ]
        
        # Process with asyncio.as_completed for progress tracking
        results = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                logger.info(f"[Job {job_id}] Progress: {completed}/{total_files} files ({completed/total_files*100:.1f}%)")
        
        logger.info(f"[Job {job_id}] All {total_files} files processed!")
        
        # Build BM25 index
        processed_files_count = sum(1 for r in results if r['status'] == 'success')
        all_texts = []
        for result in results:
            if result['status'] == 'success' and 'texts' in result:
                all_texts.extend(result['texts'])
        
        if all_texts:
            logger.info(f"[Job {job_id}] Building BM25 index from {len(all_texts)} texts...")
            await self._build_bm25_index(all_texts)
        
        failed_files = [r for r in results if r['status'] == 'failed']
        
        return {
            'status': 'completed',
            'total_files': total_files,
            'processed_successfully': processed_files_count,
            'failed_count': len(failed_files),
            'failed_files': [f['file_name'] for f in failed_files],
            'bm25_texts_count': len(all_texts),
            'processing_stats': {
                'batch_size': batch_size,
                'max_concurrent': max_concurrent
            }
        }

