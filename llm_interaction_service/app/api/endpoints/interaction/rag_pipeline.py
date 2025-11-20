import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from app.core.middleware import authorize_client
from app.factories.rag_service_factory import RagServiceFactory
from app.interfaces.rag_service_interface import RagServiceInterface
from app.core.constants import ModelName, AIService, ChromaCollection, RetrievalStrategy

router = APIRouter()
logger = logging.getLogger(__name__)

def get_rag_service() -> RagServiceInterface:
    return RagServiceFactory.create_rag_service()

@router.post("/rag-pipeline")
async def run_rag_pipeline(
    prompt: str = Form(...),
    model: str = Form(ModelName.LLAMA33.value),
    ai_service: AIService = Form(AIService.GROQ_CLOUD.value),
    collection_name: ChromaCollection = Form(ChromaCollection.RCP_DOCUMENTS_V2.value),
    retrieval_strategy: Optional[RetrievalStrategy] = Form(RetrievalStrategy.HYBRID.value),
    top_k: Optional[int] = Form(5),
    _: bool = Depends(authorize_client),
    rag_service: RagServiceInterface = Depends(get_rag_service)
):
    """
    Run the RAG pipeline for a doctor's query with hybrid retrieval support.
    
    - **prompt**: The doctor's question.
    - **model**: The LLM model to use.
    - **ai_service**: The AI service to use (e.g., 'groq_cloud').
    - **collection_name**: The ChromaDB collection to search in (default: rcp_documents_v2 with BGE embeddings).
    - **retrieval_strategy**: Retrieval strategy - 'hybrid' (default), 'vector_only', or 'bm25_only'.
    - **top_k**: Number of documents to retrieve (default: 5).
    
    **New Features**:
    - Hybrid retrieval combines vector search + BM25 keyword search + cross-encoder reranking
    - Section-aware chunking for RCP documents
    - Prompt guardrails to prevent hallucinations
    - Source citations with RCP section numbers
    """
    try:
        logger.info(
            f"RAG query: prompt='{prompt[:50]}...', strategy={retrieval_strategy}, "
            f"collection={collection_name}, top_k={top_k}"
        )
        
        result = await rag_service.query(
            model=model,
            prompt=prompt,
            ai_service=ai_service,
            collection_name=collection_name,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k
        )
        
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG pipeline error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in RAG pipeline: {str(e)}")

@router.post("/rag-pipeline/benchmark")
async def benchmark_retrieval_strategies(
    prompt: str = Form(...),
    model: str = Form(ModelName.LLAMA33.value),
    ai_service: AIService = Form(AIService.GROQ_CLOUD.value),
    collection_name: ChromaCollection = Form(ChromaCollection.RCP_DOCUMENTS_V2.value),
    _: bool = Depends(authorize_client),
    rag_service: RagServiceInterface = Depends(get_rag_service)
):
    """
    Benchmark different retrieval strategies for the same query.
    Returns results from vector_only, bm25_only, and hybrid strategies for comparison.
    
    - **prompt**: The doctor's question.
    - **model**: The LLM model to use.
    - **ai_service**: The AI service to use.
    - **collection_name**: The ChromaDB collection to search in.
    
    **Returns**: Dictionary with results from each strategy for comparison.
    """
    try:
        logger.info(f"Benchmarking retrieval strategies for prompt: '{prompt[:50]}...'")
        
        strategies = [
            RetrievalStrategy.VECTOR_ONLY.value,
            RetrievalStrategy.BM25_ONLY.value,
            RetrievalStrategy.HYBRID.value
        ]
        
        results = {}
        for strategy in strategies:
            logger.info(f"Testing strategy: {strategy}")
            result = await rag_service.query(
                model=model,
                prompt=prompt,
                ai_service=ai_service,
                collection_name=collection_name,
                retrieval_strategy=strategy,
                top_k=5
            )
            results[strategy] = {
                'response': result.get('response'),
                'num_documents': result.get('num_documents_retrieved'),
                'documents': result.get('retrieved_documents', []),
                'low_confidence': result.get('low_confidence', False)
            }
        
        return JSONResponse(content={
            'query': prompt,
            'results': results,
            'message': 'Benchmark completed for all retrieval strategies'
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in benchmark: {str(e)}")
