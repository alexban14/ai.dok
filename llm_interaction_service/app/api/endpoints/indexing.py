import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, Query, Path as PathParam
from app.core.middleware import authorize_client
from app.services.indexing_service import IndexingService
from app.services.background_job_manager import job_manager
from app.factories.indexing_service_factory import IndexingServiceFactory

router = APIRouter()
logger = logging.getLogger(__name__)

def get_indexing_service() -> IndexingService:
    return IndexingServiceFactory.create_indexing_service()

@router.post("/process-bucket")
async def process_bucket(
        max_concurrent: Optional[int] = Query(default=20, ge=1, le=100, description="Maximum concurrent file processing tasks"),
        batch_size: Optional[int] = Query(default=50, ge=5, le=200, description="Batch size for BM25 indexing"),
        _: bool = Depends(authorize_client),
        indexing_service: IndexingService = Depends(get_indexing_service)
):
    """
    Trigger the processing of the entire bucket of RCP documents with concurrent processing.
    
    **Best Practice Implementation:**
    - Returns immediately with job_id
    - Processing runs in separate process (no worker blocking)
    - Use GET /process-bucket/{job_id} to check progress
    
    **Performance Tuning:**
    - `max_concurrent`: Number of files processed simultaneously (default: 20, range: 1-100)
      - Higher = faster but more memory/CPU usage
      - Recommended: 20-50 for servers with 8+ CPU cores, 16GB+ RAM
      - Recommended: 50-100 for powerful servers (16+ cores, 32GB+ RAM)
    - `batch_size`: Batch size for BM25 index building (default: 50)
    
    **Expected Performance:**
    - Sequential (max_concurrent=1): ~30-40 seconds/file → 50-55 hours for 6000 files
    - Concurrent (max_concurrent=20): ~3-5 seconds/file → 5-8 hours for 6000 files
    - Concurrent (max_concurrent=50): ~1-2 seconds/file → 2-3.5 hours for 6000 files
    - Concurrent (max_concurrent=100): ~0.5-1 second/file → 1-2 hours for 6000 files (requires powerful hardware)
    """
    # Generate unique job ID
    job_id = f"indexing_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"Creating background job {job_id}: max_concurrent={max_concurrent}, batch_size={batch_size}")
    
    # Start job in separate process (best practice - no worker blocking!)
    status = job_manager.start_job(
        job_id=job_id,
        target_func=indexing_service.process_bucket_sync,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        job_id_param=job_id  # Pass job_id for progress updates
    )
    
    # Calculate estimated time based on max_concurrent
    if max_concurrent >= 100:
        time_estimate = "1-2 hours for 6000 files"
    elif max_concurrent >= 50:
        time_estimate = "2-3.5 hours for 6000 files"
    elif max_concurrent >= 20:
        time_estimate = "5-8 hours for 6000 files"
    else:
        time_estimate = "8-13 hours for 6000 files"
    
    return {
        "message": "Bucket processing started in background process.",
        "job_id": job_id,
        "status_url": f"/indexing/process-bucket/{job_id}",
        "config": {
            "max_concurrent": max_concurrent,
            "batch_size": batch_size,
            "estimated_time": time_estimate
        },
        "optimization_tips": [
            f"Current: max_concurrent={max_concurrent}",
            "Check progress: GET /indexing/process-bucket/{job_id}",
            "Monitor resources: docker stats ai-dok-llm_interaction_service-1"
        ]
    }


@router.get("/process-bucket/{job_id}")
async def get_job_status(
        job_id: str = PathParam(..., description="Job ID returned by POST /process-bucket"),
        _: bool = Depends(authorize_client)
):
    """
    Get status of a background processing job.
    
    Returns:
        - status: pending, running, completed, failed, cancelled
        - progress: current progress (files processed)
        - total: total files to process
        - current_file: name of file being processed
        - result: final result (when completed)
        - error: error message (when failed)
    """
    status = job_manager.get_status(job_id)
    
    return {
        "job_id": job_id,
        "status": status.status,
        "started_at": status.started_at,
        "completed_at": status.completed_at,
        "progress": {
            "current": status.progress,
            "total": status.total,
            "percentage": round(status.progress / status.total * 100, 2) if status.total > 0 else 0,
            "current_file": status.current_file
        },
        "result": status.result,
        "error": status.error
    }


@router.delete("/process-bucket/{job_id}")
async def cancel_job(
        job_id: str = PathParam(..., description="Job ID to cancel"),
        _: bool = Depends(authorize_client)
):
    """Cancel a running background job."""
    success = job_manager.cancel_job(job_id)
    
    if success:
        return {"message": f"Job {job_id} cancelled successfully"}
    else:
        return {"message": f"Job {job_id} not found or already completed"}