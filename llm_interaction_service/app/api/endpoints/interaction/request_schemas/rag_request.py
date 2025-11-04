from pydantic import BaseModel
from app.core.constants import ModelName, AIService, ChromaCollection

class RagRequest(BaseModel):
    prompt: str
    model: str = ModelName.LLAMA33.value
    ai_service: AIService = AIService.GROQ_CLOUD
    collection_name: ChromaCollection = ChromaCollection.RCP_DOCUMENTS
