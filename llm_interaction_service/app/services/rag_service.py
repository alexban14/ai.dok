import logging
import json
import re
from typing import Dict, Optional
from fastapi import HTTPException
from app.factories.llm_interaction_service_factory import LlmInteractionServiceFactory
from app.factories.vector_store_service_factory import VectorStoreServiceFactory
from app.factories.hybrid_retrieval_service_factory import HybridRetrievalServiceFactory
from app.interfaces.rag_service_interface import RagServiceInterface
from app.core.config import config
from app.core.constants import RetrievalStrategy

logger = logging.getLogger(__name__)

class RagService(RagServiceInterface):
    def __init__(
        self,
        ollama_base_url: str = None,
        groq_api_key: str = None,
        collection_name: str = None,
        retrieval_strategy: str = None
    ):
        self.collection_name = collection_name
        self.retrieval_strategy = retrieval_strategy or config.retrieval_strategy
        
        # Initialize services
        self.vector_store_service = VectorStoreServiceFactory.create_vector_store_service(
            collection_name=self.collection_name
        )
        
        # Initialize hybrid retrieval service
        self.hybrid_retrieval_service = HybridRetrievalServiceFactory.create_hybrid_retrieval_service(
            vector_store=self.vector_store_service,
            collection_name=self.collection_name
        )
        
        self.ollama_base_url = ollama_base_url or config.ollama_base_url
        self.groq_api_key = groq_api_key or config.groq_api_key
        self._llm_service = None
        
        logger.info(
            f"RAG Service initialized: collection={self.collection_name}, "
            f"strategy={self.retrieval_strategy}"
        )

    def _create_prompt(self, retrieved_text: str, user_prompt: str, retrieved_sources: list = None) -> Dict[str, str]:
        """
        Create a prompt for the LLM with explicit guardrails to prevent hallucinations.
        
        Args:
            retrieved_text: Context from retrieved documents
            user_prompt: User's question
            retrieved_sources: List of source documents with metadata
            
        Returns:
            Dictionary with system and user prompts
        """
        # Build source citations
        source_info = ""
        if retrieved_sources:
            unique_sources = set()
            for doc in retrieved_sources:
                metadata = doc.get('metadata', {})
                source = metadata.get('source', 'Unknown')
                section = metadata.get('section_number', '')
                section_title = metadata.get('section_title', '')
                
                if section and section_title:
                    unique_sources.add(f"- {source} (Secțiunea {section}: {section_title})")
                elif section:
                    unique_sources.add(f"- {source} (Secțiunea {section})")
                else:
                    unique_sources.add(f"- {source}")
            
            if unique_sources:
                source_info = "\n\nDocumente sursa:\n" + "\n".join(sorted(unique_sources))
        
        system = f"""
System:
You are an AI medical assistant specialized in analyzing RCP (Rezumat Caracteristici Produsului) documents for healthcare professionals.

CRITICAL RULES - YOU MUST FOLLOW THESE STRICTLY:

1. **ANSWER ONLY FROM PROVIDED CONTEXT**: You must base your response EXCLUSIVELY on the retrieved context below. Do NOT use external knowledge or make assumptions.

2. **IF INFORMATION IS NOT AVAILABLE**: If the context does not contain sufficient information to answer the question, you MUST respond:
   "Nu există informații suficiente în documentele RCP disponibile pentru a răspunde la această întrebare."

3. **CITE SOURCES**: When providing information, always cite the specific RCP section (e.g., "Conform secțiunii 4.1 (Indicații terapeutice)...").

4. **BE PRECISE AND FACTUAL**: Use exact dosages, contraindications, and medical terms from the context. Do not paraphrase in ways that could alter meaning.

5. **MEDICAL SAFETY**: For critical information (dosages, contraindications, adverse reactions), quote directly from the source text.

6. **FORMAT YOUR RESPONSE AS HTML**: The output must be valid HTML that can be displayed in a web application using innerHTML. Use:
   - <h3> for section headings
   - <ul> and <li> for lists
   - <strong> for emphasis
   - <p> for paragraphs
   - <div class="warning"> for important warnings

Retrieved Context from RCP documents:
{retrieved_text}
{source_info}

REMEMBER: If the answer is not in the context above, say "Nu există informații suficiente" instead of guessing or using external knowledge.
        """
        
        user = f"""
User Question: {user_prompt}

Please provide a comprehensive answer in Romanian, formatted as HTML, based ONLY on the retrieved context above.

**IMPORTANT**: Return your response as a JSON object with the following structure:
{{
  "response": "Your HTML-formatted answer here"
}}
        """
        
        return {"system": system, "user": user}

    async def query(
        self,
        model: str,
        prompt: str,
        ai_service: str,
        collection_name: str = None,
        retrieval_strategy: str = None,
        top_k: int = None
    ) -> Dict:
        """
        Query RAG system with hybrid retrieval support.
        
        Args:
            model: LLM model name
            prompt: User query
            ai_service: AI service provider
            collection_name: ChromaDB collection (optional override)
            retrieval_strategy: Retrieval strategy (vector_only/hybrid/bm25_only)
            top_k: Number of documents to retrieve (optional override)
            
        Returns:
            Dictionary with response and metadata
        """
        # Initialize LLM service
        self._llm_service = LlmInteractionServiceFactory.create_llm_interaction_service(
            ai_service,
            self.ollama_base_url,
            self.groq_api_key
        )
        
        # Use provided or default values
        strategy = retrieval_strategy or self.retrieval_strategy
        k = top_k or config.reranker_top_k
        
        try:
            # 1. Retrieve documents using specified strategy
            logger.info(
                f"Performing {strategy} retrieval in collection '{self.collection_name}' "
                f"for prompt: '{prompt}'"
            )
            
            if strategy == RetrievalStrategy.VECTOR_ONLY.value:
                # Legacy vector-only retrieval
                retrieved_docs = self.vector_store_service.similarity_search(query=prompt, k=k)
                retrieved_docs_with_scores = [(doc, 1.0) for doc in retrieved_docs]
            else:
                # Hybrid or BM25 retrieval
                retrieved_docs_with_scores = self.hybrid_retrieval_service.retrieve(
                    query=prompt,
                    strategy=strategy,
                    k=k
                )
            
            # Extract documents and scores
            retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
            scores = [score for doc, score in retrieved_docs_with_scores]
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using {strategy} strategy")
            
            # Build context text
            retrieved_text = "\n\n---\n\n".join([
                f"[Document {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Prepare document metadata for citations
            retrieved_docs_metadata = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(scores[i]) if i < len(scores) else 0.0
                }
                for i, doc in enumerate(retrieved_docs)
            ]
            
            # 2. Create prompt with guardrails
            llm_prompt = self._create_prompt(
                retrieved_text,
                prompt,
                retrieved_sources=retrieved_docs_metadata
            )
            
            logger.debug(f"Prompt created with {len(retrieved_text)} chars of context")

            # 3. Generate completion
            logger.info(f"Generating completion with {ai_service} using model {model}")
            result = ""
            async for chunk in self._llm_service.generate_completion(
                    model=model,
                    prompt=llm_prompt,
                    stream=False
            ):
                result += chunk["response"]

            # 4. Clean and parse response
            cleaned_text = re.sub(r'^```json\n|\n```$', "", result).strip()
            logger.debug(f"LLM raw response: {result[:200]}...")

            try:
                llm_response = json.loads(cleaned_text)
            except json.JSONDecodeError:
                logger.warning("LLM response is not valid JSON. Wrapping in standard format.")
                llm_response = {"response": cleaned_text}

            # 5. Add metadata to response
            llm_response.update({
                'retrieved_documents': retrieved_docs_metadata,
                'retrieval_strategy': strategy,
                'num_documents_retrieved': len(retrieved_docs),
                'collection_name': self.collection_name,
                'query': prompt
            })
            
            # 6. Check for hallucination indicators
            response_text = llm_response.get('response', '').lower()
            hallucination_indicators = [
                'nu există informații',
                'nu am găsit',
                'informații insuficiente',
                'nu pot găsi',
                'nu este disponibil'
            ]
            llm_response['low_confidence'] = any(
                indicator in response_text
                for indicator in hallucination_indicators
            )

            return llm_response

        except Exception as e:
            logger.error(f"Error during RAG query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during RAG query: {str(e)}")
