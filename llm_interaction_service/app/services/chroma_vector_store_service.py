import logging
from typing import List
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from app.core.config import config
from app.core.constants import ChromaCollection
from app.interfaces.vector_store_service_interface import VectorStoreServiceInterface

logger = logging.getLogger(__name__)

class ChromaVectorStoreService(VectorStoreServiceInterface):
    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_client = chromadb.HttpClient(host=config.chroma_db_host, port=config.chroma_db_port)
        self.client = Chroma(
            collection_name=ChromaCollection.RCP_DOCUMENTS.value,
            client=self.db_client,
            embedding_function=self.embedding_function
        )

    def add_texts(self, texts: List[str], metadatas: List[dict], ids: List[str] = None):
        self.client.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, k: int) -> List[Document]:
        return self.client.similarity_search(query=query, k=k)

    def get_collection(self, collection_name: str) -> dict:
        collection = self.db_client.get_collection(name=collection_name)
        return collection.get()

    def delete_collection(self, collection_name: str) -> None:
        self.db_client.delete_collection(name=collection_name)