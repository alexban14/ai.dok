import logging
from typing import List, Optional
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from app.core.config import config
from app.core.constants import ChromaCollection
from app.interfaces.vector_store_service_interface import VectorStoreServiceInterface

logger = logging.getLogger(__name__)

# Global embedding model cache - loaded once per process, not per request!
_embedding_model_cache = {}

class ChromaVectorStoreService(VectorStoreServiceInterface):
    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize ChromaDB vector store with configurable collection and embedding model.
        Uses global model cache to avoid reloading 2.5GB model on every request.
        
        Args:
            collection_name: Name of collection (default: RCP_DOCUMENTS_V2 for new BGE embeddings)
            embedding_model: Embedding model name (default: BGE-Large from config)
        """
        # Use new collection and BGE model by default
        self.collection_name = collection_name or ChromaCollection.RCP_DOCUMENTS_V2.value
        self.embedding_model_name = embedding_model or config.embedding_model
        
        logger.info(f"Initializing ChromaDB with collection '{self.collection_name}' and model '{self.embedding_model_name}'")
        
        # Use cached embedding model to avoid reloading (MAJOR optimization!)
        if self.embedding_model_name not in _embedding_model_cache:
            logger.info(f"Loading embedding model {self.embedding_model_name} (first time, ~2.5GB)...")
            _embedding_model_cache[self.embedding_model_name] = SentenceTransformerEmbeddings(
                model_name=self.embedding_model_name
            )
            logger.info(f"Embedding model loaded and cached")
        else:
            logger.info(f"Using cached embedding model {self.embedding_model_name}")
        
        self.embedding_function = _embedding_model_cache[self.embedding_model_name]
        
        # Initialize ChromaDB client
        self.db_client = chromadb.HttpClient(host=config.chroma_db_host, port=config.chroma_db_port)
        
        # Initialize Chroma vector store
        self.client = Chroma(
            collection_name=self.collection_name,
            client=self.db_client,
            embedding_function=self.embedding_function
        )
        
        logger.info(f"ChromaDB initialized successfully")

    def add_texts(self, texts: List[str], metadatas: List[dict], ids: List[str] = None):
        """
        Add texts to vector store with optimized batching.
        Larger batches reduce HTTP round-trips to ChromaDB.
        """
        batch_size = 500  # Increased from 100 - fewer HTTP calls
        
        if len(texts) <= batch_size:
            # Single batch - most common case for individual files
            self.client.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        else:
            # Large batch (rare), split efficiently
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size] if ids else None
                
                self.client.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

    def similarity_search(self, query: str, k: int) -> List[Document]:
        return self.client.similarity_search(query=query, k=k)

    def get_collection(self, collection_name: str) -> dict:
        collection = self.db_client.get_collection(name=collection_name)
        return collection.get()

    def delete_collection(self, collection_name: str) -> None:
        self.db_client.delete_collection(name=collection_name)
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents from the current collection.
        Useful for building BM25 index.
        
        Returns:
            List of all Document objects in collection
        """
        try:
            collection = self.db_client.get_collection(name=self.collection_name)
            results = collection.get(include=['documents', 'metadatas'])
            
            documents = []
            for i, doc_text in enumerate(results['documents']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))
            
            logger.info(f"Retrieved {len(documents)} documents from collection '{self.collection_name}'")
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve all documents: {e}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of collection to check
            
        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.db_client.list_collections()
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False