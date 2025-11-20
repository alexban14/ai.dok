import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from langchain.docstore.document import Document
from app.services.bm25_service import BM25Service
from app.services.reranker_service import RerankerService
from app.interfaces.vector_store_service_interface import VectorStoreServiceInterface
from app.core.config import config

logger = logging.getLogger(__name__)

class HybridRetrievalService:
    """
    Hybrid retrieval combining vector search (semantic) + BM25 (keyword) + reranking.
    Uses Reciprocal Rank Fusion (RRF) to combine results from different retrievers.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreServiceInterface,
        bm25_service: BM25Service,
        reranker_service: RerankerService,
        alpha: float = None,
        retrieval_top_k: int = None,
        reranker_top_k: int = None
    ):
        """
        Initialize hybrid retrieval service.
        
        Args:
            vector_store: Vector database service
            bm25_service: BM25 keyword search service
            reranker_service: Cross-encoder reranker
            alpha: Weight for combining scores (0=BM25 only, 1=vector only)
            retrieval_top_k: Number of candidates to retrieve initially
            reranker_top_k: Final number of results after reranking
        """
        self.vector_store = vector_store
        self.bm25_service = bm25_service
        self.reranker_service = reranker_service
        
        # Configuration
        self.alpha = alpha if alpha is not None else config.hybrid_alpha
        self.retrieval_top_k = retrieval_top_k if retrieval_top_k is not None else config.retrieval_top_k
        self.reranker_top_k = reranker_top_k if reranker_top_k is not None else config.reranker_top_k
        
        logger.info(
            f"Hybrid Retrieval initialized: alpha={self.alpha}, "
            f"retrieval_top_k={self.retrieval_top_k}, reranker_top_k={self.reranker_top_k}"
        )
    
    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using specified strategy.
        
        Args:
            query: Search query
            strategy: Retrieval strategy ("hybrid", "vector_only", "bm25_only")
            k: Number of final results (default: reranker_top_k)
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        k = k if k is not None else self.reranker_top_k
        
        if strategy == "vector_only":
            return self._vector_only_retrieval(query, k)
        elif strategy == "bm25_only":
            return self._bm25_only_retrieval(query, k)
        else:  # hybrid
            return self._hybrid_retrieval(query, k)
    
    def _vector_only_retrieval(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        Retrieve using vector search only.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        logger.debug(f"Vector-only retrieval for query: '{query}'")
        
        # Get documents from vector store
        documents = self.vector_store.similarity_search(query, k=k)
        
        # Return with placeholder scores (vector store doesn't return scores in this interface)
        return [(doc, 1.0) for doc in documents]
    
    def _bm25_only_retrieval(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        Retrieve using BM25 only.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        logger.debug(f"BM25-only retrieval for query: '{query}'")
        
        # Get all documents from vector store (needed for BM25 mapping)
        # This is a limitation - we need access to the full corpus
        # In production, maintain separate document store
        all_docs = self.vector_store.similarity_search(query, k=100)
        
        if not all_docs:
            logger.warning("No documents available for BM25 search")
            return []
        
        # Search BM25
        bm25_results = self.bm25_service.search_documents(query, all_docs, k=k)
        
        return bm25_results
    
    def _hybrid_retrieval(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        Hybrid retrieval using RRF (Reciprocal Rank Fusion).
        
        Args:
            query: Search query
            k: Final number of results
            
        Returns:
            List of (Document, score) tuples
        """
        logger.debug(f"Hybrid retrieval for query: '{query}'")
        
        # Step 1: Get candidates from vector search
        vector_docs = self.vector_store.similarity_search(query, k=self.retrieval_top_k)
        
        if not vector_docs:
            logger.warning("No documents found in vector search")
            return []
        
        # Step 2: Get candidates from BM25
        bm25_results = self.bm25_service.search_documents(query, vector_docs, k=self.retrieval_top_k)
        bm25_docs = [doc for doc, score in bm25_results]
        
        # Step 3: Combine using Reciprocal Rank Fusion
        combined_docs = self._reciprocal_rank_fusion(
            vector_docs,
            bm25_docs,
            top_k=min(self.retrieval_top_k, len(vector_docs))
        )
        
        if not combined_docs:
            logger.warning("No documents after RRF fusion")
            return []
        
        # Step 4: Rerank top candidates
        reranked_results = self.reranker_service.rerank_documents(
            query,
            combined_docs,
            top_k=k
        )
        
        logger.info(f"Hybrid retrieval returned {len(reranked_results)} documents")
        return reranked_results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
        k: int = 60,
        top_k: int = 20
    ) -> List[Document]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = sum_over_retrievers(1 / (k + rank(d)))
        
        Args:
            vector_results: Documents from vector search (ordered by relevance)
            bm25_results: Documents from BM25 search (ordered by relevance)
            k: Constant for RRF (default 60)
            top_k: Number of results to return
            
        Returns:
            List of Documents sorted by fused score
        """
        # Create document ID mapping (use page_content hash as ID)
        doc_scores = {}
        doc_objects = {}
        
        # Process vector results
        for rank, doc in enumerate(vector_results):
            doc_id = id(doc)  # Use object id as unique identifier
            doc_objects[doc_id] = doc
            # RRF score weighted by alpha
            doc_scores[doc_id] = self.alpha / (k + rank + 1)
        
        # Process BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = id(doc)
            doc_objects[doc_id] = doc
            # RRF score weighted by (1-alpha)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (1 - self.alpha) / (k + rank + 1)
        
        # Sort by fused score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Return top-k documents
        combined_docs = [doc_objects[doc_id] for doc_id in sorted_doc_ids[:top_k]]
        
        logger.debug(f"RRF fused {len(vector_results)} vector + {len(bm25_results)} BM25 results")
        return combined_docs
    
    def retrieve_with_metadata(
        self,
        query: str,
        strategy: str = "hybrid",
        k: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Retrieve documents with detailed metadata about retrieval process.
        
        Args:
            query: Search query
            strategy: Retrieval strategy
            k: Number of results
            
        Returns:
            Dictionary with results and metadata
        """
        k = k if k is not None else self.reranker_top_k
        
        # Retrieve documents
        results = self.retrieve(query, strategy, k)
        
        # Build response with metadata
        return {
            'query': query,
            'strategy': strategy,
            'results': [
                {
                    'document': doc,
                    'score': score,
                    'metadata': doc.metadata
                }
                for doc, score in results
            ],
            'total_results': len(results),
            'parameters': {
                'alpha': self.alpha,
                'retrieval_top_k': self.retrieval_top_k,
                'reranker_top_k': self.reranker_top_k
            }
        }
