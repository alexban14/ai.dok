import logging
from fastapi import APIRouter, Depends
from app.core.middleware import authorize_client
from app.factories.vector_store_service_factory import VectorStoreServiceFactory
from app.interfaces.vector_store_service_interface import VectorStoreServiceInterface

router = APIRouter()
logger = logging.getLogger(__name__)

def get_vector_store_service() -> VectorStoreServiceInterface:
    return VectorStoreServiceFactory.create_vector_store_service()

@router.get("/{collection_name}")
def get_collection(collection_name: str, client_id: int, authorized: bool = Depends(authorize_client), vector_store_service: VectorStoreServiceInterface = Depends(get_vector_store_service)):
    """
    Get all texts from a collection.
    """
    return vector_store_service.get_collection(collection_name)

@router.delete("/{collection_name}")
def delete_collection(collection_name: str, client_id: int, authorized: bool = Depends(authorize_client), vector_store_service: VectorStoreServiceInterface = Depends(get_vector_store_service)):
    """
    Delete a collection.
    """
    vector_store_service.delete_collection(collection_name)
    return {"message": f"Collection '{collection_name}' deleted."}
