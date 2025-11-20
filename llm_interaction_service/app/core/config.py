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

    # Hybrid Retrieval Configuration
    embedding_model: str = Field(default='BAAI/bge-large-en-v1.5', alias='EMBEDDING_MODEL')
    reranker_model: str = Field(default='BAAI/bge-reranker-large', alias='RERANKER_MODEL')
    retrieval_strategy: str = Field(default='hybrid', alias='RETRIEVAL_STRATEGY')
    
    # BM25 Parameters
    bm25_k1: float = Field(default=1.5, alias='BM25_K1')
    bm25_b: float = Field(default=0.75, alias='BM25_B')
    
    # Hybrid Retrieval Parameters
    hybrid_alpha: float = Field(default=0.5, alias='HYBRID_ALPHA')  # Weight for vector vs BM25
    retrieval_top_k: int = Field(default=20, alias='RETRIEVAL_TOP_K')  # Initial retrieval count
    reranker_top_k: int = Field(default=5, alias='RERANKER_TOP_K')  # Final results after reranking
    
    # RCP Section Chunking
    chunk_by_section: bool = Field(default=True, alias='CHUNK_BY_SECTION')
    chunk_size: int = Field(default=512, alias='CHUNK_SIZE')
    chunk_overlap: int = Field(default=100, alias='CHUNK_OVERLAP')

    client_ids: List[int] = Field(..., alias='CLIENT_IDS')
    api_access_tokens: List[str] = Field(..., alias='API_ACCESS_TOKENS')

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', populate_by_name=True)


config = Config()