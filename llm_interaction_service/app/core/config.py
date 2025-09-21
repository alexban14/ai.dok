from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Config(BaseSettings):
    service_name: str = 'service_name'
    secret_key: str = 's3cr3t_k3y'
    ocr_processing_service: str = Field(..., alias='OCR_PROCESSING_SERVICE')
    pdf_to_image_service: str = Field(..., alias='PDF_TO_IMAGE_SERVICE')
    ollama_base_url: str = Field(..., alias='OLLAMA_BASE_URL')
    groq_api_key: str = Field(..., alias='GROQ_API_KEY')

    b2_bucket_name: str = Field(..., alias='B2_BUCKET_NAME')
    b2_application_key_id: str = Field(..., alias='B2_APPLICATION_KEY_ID')
    b2_application_key: str = Field(..., alias='B2_APPLICATION_KEY')

    chroma_db_host: str = Field(..., alias='CHROMA_DB_HOST')
    chroma_db_port: int = Field(..., alias='CHROMA_DB_PORT')

    client_ids: List[int] = Field(..., alias='CLIENT_IDS')
    api_access_tokens: List[str] = Field(..., alias='API_ACCESS_TOKENS')

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', populate_by_name=True)


config = Config()