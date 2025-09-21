import logging
from typing import List
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from app.core.config import config
from app.interfaces.vector_store_service_interface import VectorStoreServiceInterface

logger = logging.getLogger(__name__)

class ChromaVectorStoreService(VectorStoreServiceInterface):
    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.client = Chroma(
            client=chromadb.HttpClient(host=config.chroma_db_host, port=config.chroma_db_port),
            embedding_function=self.embedding_function
        )

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        self.client.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int) -> List[Document]:
        return self.client.similarity_search(query=query, k=k)