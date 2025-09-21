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
    MISTRAL = "mistral"
    # Groq models
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    LLAMA2_70B = "llama-3.3-70b-versatile"