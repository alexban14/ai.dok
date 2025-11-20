import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from langchain.docstore.document import Document
from app.core.config import config

logger = logging.getLogger(__name__)

class RerankerService:
    """
    Service for reranking retrieved documents using cross-encoder.
    Cross-encoders provide higher accuracy than bi-encoders for ranking.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize reranker with cross-encoder model.
        
        Args:
            model_name: HuggingFace model name (default from config)
        """
        self.model_name = model_name or config.reranker_model
        logger.info(f"Initializing reranker with model: {self.model_name}")
        
        try:
            self.model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of (document_index, score) tuples sorted by relevance
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [(i, 0.0) for i in range(min(top_k, len(documents)))]
        
        # Sort by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Return top-k with scores
        results = [
            (int(idx), float(scores[idx])) 
            for idx in ranked_indices[:top_k]
        ]
        
        logger.debug(f"Reranked {len(documents)} documents, returning top {len(results)}")
        return results
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Rerank Document objects.
        
        Args:
            query: Search query
            documents: List of Document objects
            top_k: Number of top results
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        
        # Rerank
        ranked_indices = self.rerank(query, texts, top_k)
        
        # Map back to Document objects
        results = [
            (documents[idx], score)
            for idx, score in ranked_indices
        ]
        
        return results
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """
        Rerank multiple query-document sets in batch.
        
        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            top_k: Number of results per query
            
        Returns:
            List of reranked results (one per query)
        """
        results = []
        for query, documents in zip(queries, documents_list):
            ranked = self.rerank(query, documents, top_k)
            results.append(ranked)
        
        return results
