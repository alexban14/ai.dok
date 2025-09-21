import os
import logging
import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from app.core.config import config
from app.services.parse_file_service import ParseFileService
from app.core.constants import ProcessingType
from app.core.constants import AIService
from app.core.constants import OCRService
from app.core.constants import ModelName
from app.core.middleware import authorize_client

router = APIRouter()
logger = logging.getLogger(__name__)

# Dependency to provide LlmInteractionService
def get_llm_interaction_service() -> ParseFileService:
    ollama_base_url = config.ollama_base_url
    groq_api_key = config.groq_api_key
    return ParseFileService(ollama_base_url=ollama_base_url, groq_api_key=groq_api_key)

@router.post("/process-file")
async def process_rcp(
        _: bool = Depends(authorize_client),
        model: str = Form(...),
        file: UploadFile = File(...),
        processing_type: str = Form(...),
        prompt: str = Form(None),
        ai_service: str = Form(AIService.GROQ_CLOUD),
        ocr_technology: str = Form(OCRService.PADDLE),
        parse_file_service: ParseFileService = Depends(get_llm_interaction_service)
):
    """
    Process RCP PDF with the specified AI service.

    - "ai_service": Choose between "ollama_local" or "groq_cloud"
    - "processing_type":
        - "parse": Extracts structured RCP details.
        - "prompt": Sends extracted file text to the LLM for a custom response.
    """
    try:
        if ai_service not in ["ollama_local", "groq_cloud"]:
            raise HTTPException(status_code=400, detail="Invalid AI service. Use 'ollama_local' or 'groq_cloud'.")

        result = await parse_file_service.process(
            model=model,
            file=file,
            processing_type=processing_type,
            prompt=prompt,
            ai_service=ai_service
        )

        return JSONResponse(content=result)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")