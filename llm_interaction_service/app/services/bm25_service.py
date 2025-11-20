import logging
import pickle
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class BM25Service:
    """
    Service for BM25 keyword-based retrieval.
    Provides fast keyword matching for medical terminology.
    """
    
    def __init__(self, corpus: Optional[List[str]] = None, index_path: Optional[str] = None):
        """
        Initialize BM25 service with corpus or load from index.
        
        Args:
            corpus: List of text documents to index
            index_path: Path to saved BM25 index
        """
        self.bm25 = None
        self.corpus = []
        self.tokenized_corpus = []
        self.index_path = index_path
        
        if corpus:
            self.build_index(corpus)
        elif index_path and Path(index_path).exists():
            self.load_index(index_path)
    
    def build_index(self, corpus: List[str]) -> None:
        """
        Build BM25 index from corpus.
        
        Args:
            corpus: List of text documents
        """
        logger.info(f"Building BM25 index for {len(corpus)} documents")
        self.corpus = corpus
        
        # Tokenize corpus (simple whitespace + lowercase)
        # For medical text, consider keeping case and adding domain-specific tokenization
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built successfully")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        Simple whitespace tokenization with lowercase.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # For medical text, we want to preserve:
        # - Numbers (dosages: "50 mg", "5-Fluorouracil")
        # - Special chars in compound names (COX-1, COX-2)
        # - Section numbers (4.1, 4.2)
        
        # Simple tokenization: split on whitespace, lowercase
        tokens = text.lower().split()
        return tokens
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search BM25 index for query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Return (index, score) pairs
        results = [(int(idx), float(scores[idx])) for idx in top_k_indices if scores[idx] > 0]
        
        logger.debug(f"BM25 search for '{query}': found {len(results)} results")
        return results
    
    def search_documents(self, query: str, documents: List[Document], k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search and return Document objects with scores.
        
        Args:
            query: Search query
            documents: List of Document objects
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        if len(documents) != len(self.corpus):
            logger.warning("Document list length doesn't match corpus length")
            return []
        
        results = self.search(query, k)
        return [(documents[idx], score) for idx, score in results]
    
    def save_index(self, path: str) -> None:
        """
        Save BM25 index to disk.
        
        Args:
            path: File path to save index
        """
        if not self.bm25:
            logger.error("No BM25 index to save")
            return
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'bm25': self.bm25,
            'corpus': self.corpus,
            'tokenized_corpus': self.tokenized_corpus
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"BM25 index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """
        Load BM25 index from disk.
        
        Args:
            path: File path to load index from
        """
        if not Path(path).exists():
            logger.error(f"BM25 index file not found: {path}")
            return
        
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.bm25 = index_data['bm25']
        self.corpus = index_data['corpus']
        self.tokenized_corpus = index_data['tokenized_corpus']
        
        logger.info(f"BM25 index loaded from {path} ({len(self.corpus)} documents)")
    
    def update_index(self, new_documents: List[str]) -> None:
        """
        Update BM25 index with new documents.
        Note: This rebuilds the entire index.
        
        Args:
            new_documents: List of new documents to add
        """
        logger.info(f"Updating BM25 index with {len(new_documents)} new documents")
        self.corpus.extend(new_documents)
        self.build_index(self.corpus)
    
    def get_corpus_size(self) -> int:
        """Get the number of documents in the index."""
        return len(self.corpus)
