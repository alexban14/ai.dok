import logging
from fastapi import APIRouter, Depends, BackgroundTasks
from app.core.middleware import authorize_client
from app.services.indexing_service import IndexingService
from app.factories.indexing_service_factory import IndexingServiceFactory

router = APIRouter()
logger = logging.getLogger(__name__)

def get_indexing_service() -> IndexingService:
    return IndexingServiceFactory.create_indexing_service()

@router.post("/process-bucket")
async def process_bucket(
        background_tasks: BackgroundTasks,
        _: bool = Depends(authorize_client),
        indexing_service: IndexingService = Depends(get_indexing_service)
):
    """
    Trigger the processing of the entire bucket of RCP documents in the background.
    """
    background_tasks.add_task(indexing_service.process_bucket)
    return {"message": "Bucket processing started in the background."}