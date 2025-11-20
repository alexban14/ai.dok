from enum import Enum

class ProcessingType(str, Enum):
    """Enum for processing types"""
    PARSE = "parse"
    PROMPT = "prompt"

class AIService(str, Enum):
    """Enum for AI service providers"""
    OLLAMA_LOCAL = "ollama_local"
    GROQ_CLOUD = "groq_cloud"

class OCRService(str, Enum):
    """Enum for OCR service providers"""
    PADDLE = "paddle"

class PDFToImageService(str, Enum):
    """Enum for PDF to image conversion services"""
    PYMUPDF_OPENCV_PILLOW = "pymupdf_opencv_pillow"

class ModelName(str, Enum):
    """Enum for supported model names"""
    # Ollama models
    LLAMA2 = "llama2"
    LLAMA33 = "llama-3.3-70b-versatile"
    MISTRAL = "mistral"
    # Groq models
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    LLAMA2_70B = "llama-3.3-70b-versatile"

class BucketProvider(str, Enum):
    B2 = "b2"

class VectorStoreProvider(str, Enum):
    CHROMA = "chroma"

class ChromaCollection(str, Enum):
    RCP_DOCUMENTS = "rcp_documents"
    RCP_DOCUMENTS_V2 = "rcp_documents_v2"

class RetrievalStrategy(str, Enum):
    """Enum for retrieval strategies"""
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    BM25_ONLY = "bm25_only"

class EmbeddingModel(str, Enum):
    """Enum for embedding models"""
    MINILM_L6_V2 = "all-MiniLM-L6-v2"
    BGE_LARGE_EN_V1_5 = "BAAI/bge-large-en-v1.5"

class RerankerModel(str, Enum):
    """Enum for reranker models"""
    BGE_RERANKER_LARGE = "BAAI/bge-reranker-large"
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"