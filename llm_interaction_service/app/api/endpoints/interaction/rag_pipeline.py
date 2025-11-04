import logging
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from app.core.middleware import authorize_client
from app.factories.rag_service_factory import RagServiceFactory
from app.interfaces.rag_service_interface import RagServiceInterface
from app.core.constants import ModelName, AIService, ChromaCollection

router = APIRouter()
logger = logging.getLogger(__name__)

def get_rag_service() -> RagServiceInterface:
    return RagServiceFactory.create_rag_service()

@router.post("/rag-pipeline")
async def run_rag_pipeline(
    prompt: str = Form(...),
    model: str = Form(ModelName.LLAMA33.value),
    ai_service: AIService = Form(AIService.GROQ_CLOUD.value),
    collection_name: ChromaCollection = Form(ChromaCollection.RCP_DOCUMENTS.value),
    _: bool = Depends(authorize_client),
    rag_service: RagServiceInterface = Depends(get_rag_service)
):
    """
    Run the RAG pipeline for a doctor's query.
    - **prompt**: The doctor's question.
    - **model**: The LLM model to use.
    - **ai_service**: The AI service to use (e.g., 'groq_cloud').
    - **collection_name**: The ChromaDB collection to search in.
    """
    try:
        result = await rag_service.query(
            model=model,
            prompt=prompt,
            ai_service=ai_service,
            collection_name=collection_name,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in RAG pipeline: {str(e)}")
