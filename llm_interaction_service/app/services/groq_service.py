import os
from typing import AsyncGenerator, Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import AIMessage
from app.interfaces.llm_interaction_service_interface import LlmInteractionServiceInterface

class GroqService(LlmInteractionServiceInterface):
    def __init__(self, api_key: str = None, timeout: int = 30):
        """
        Initialize the GroqService using Langchain's ChatGroq wrapper.

        Args:
            api_key (str, optional): Groq API key. Defaults to environment variable.
            timeout (int): Timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required")

        os.environ["GROQ_API_KEY"] = self.api_key
        self.timeout = timeout

    async def generate_completion(
        self,
        model: str,
        prompt: Dict[str, str],
        stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Uses Langchain's ChatGroq wrapper to generate text.

        Args:
            model (str): The model to use.
            prompt (str): The input prompt.
            stream (bool): If True, stream responses.

        Yields:
            dict: JSON response from Langchain's ChatGroq model.
        """
        try:
            llm = ChatGroq(
                model=model,
                temperature=0,
                max_retries=2
            ).with_structured_output(method="json_mode", include_raw=True)

            # Split the prompt into system and user messages
            response = llm.invoke([
                ("system", prompt['system']),
                ("user", prompt['user'])
            ])

            response_content = response['raw'].content if isinstance(response['raw'], AIMessage) else str(response)

            # For consistency with Ollama service
            yield {"response": response_content}

        except Exception as e:
            raise RuntimeError(f"Error in GroqService: {str(e)}") from e