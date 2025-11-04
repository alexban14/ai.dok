import logging
import json
import re
from typing import Dict
from fastapi import HTTPException
from app.factories.llm_interaction_service_factory import LlmInteractionServiceFactory
from app.factories.vector_store_service_factory import VectorStoreServiceFactory
from app.interfaces.rag_service_interface import RagServiceInterface
from app.core.config import config

logger = logging.getLogger(__name__)

class RagService(RagServiceInterface):
    def __init__(self, ollama_base_url: str = None, groq_api_key: str = None):
        self.vector_store_service = VectorStoreServiceFactory.create_vector_store_service()
        self.ollama_base_url = ollama_base_url or config.ollama_base_url
        self.groq_api_key = groq_api_key or config.groq_api_key
        self._llm_service = None

    def _create_prompt(self, retrieved_text: str, user_prompt: str) -> Dict[str, str]:
        """Create a prompt for the LLM."""
        system = f"""
            System:
            You are an AI medical assistant. Your role is to provide concise and accurate information to doctors based on official medical documents (RCPs).
            Analyze the provided text from the RCP document(s) to answer the user's question.
            The returned data should be of type JSON blob. The JSON must have a single key called "response", under that key,
            the value should be a string representing html format so that it can be displayed in a web app using "innerHTML".

            Retrieved Context from knowledge base:
            {retrieved_text}
        """
        user = f"""
            User Prompt: {user_prompt}
        """
        return {"system": system, "user": user}

    async def query(self, model: str, prompt: str, ai_service: str, collection_name: str) -> Dict:
        self._llm_service = LlmInteractionServiceFactory.create_llm_interaction_service(
            ai_service,
            self.ollama_base_url,
            self.groq_api_key
        )

        try:
            # 1. Similarity search
            logger.info(f" \n -------- \n Performing similarity search in collection '{collection_name}' for prompt: '{prompt}' \n -------- \n ")
            retrieved_docs = self.vector_store_service.similarity_search(query=prompt, k=5)
            retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            logger.info(f" \n -------- \n Retrieved {len(retrieved_docs)} documents from vector store. \n -------- \n ")

            # 2. Create prompt for LLM
            llm_prompt = self._create_prompt(retrieved_text, prompt)
            logger.info(f"'{llm_prompt}'")

            logger.info(f" \n -------- \n Prompt for LLM Service: {llm_prompt} \n -------- \n ")

            # 3. Generate completion
            logger.info(f" \n -------- \n Generating completion with {ai_service} using model {model}. \n -------- \n ")
            result = ""
            async for chunk in self._llm_service.generate_completion(
                    model=model,
                    prompt=llm_prompt,
                    stream=False
            ):
                result += chunk["response"]

            # 4. Clean and parse response
            cleaned_text = re.sub(r'^```json\n|\n```$', "", result).strip()
            logger.debug(f" \n -------- \n LLM raw response: {result} \n -------- \n ")
            logger.debug(f" \n -------- \n LLM cleaned response: {cleaned_text} \n -------- \n ")

            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                logger.warning("LLM response is not a valid JSON. Returning as string.")
                return {"response": cleaned_text}

        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
            raise HTTPException(status_code=500, detail=f"Error during RAG query: {str(e)}")
