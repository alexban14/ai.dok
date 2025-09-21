from langchain_community.llms import Ollama
from typing import AsyncGenerator, Dict, Any
from app.interfaces.llm_interaction_service_interface import LlmInteractionServiceInterface

class OllamaService(LlmInteractionServiceInterface):
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the OllamaService using Langchain's Ollama wrapper.

        Args:
            base_url (str): The base URL for the Ollama API.
            timeout (int): Timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate_completion(
            self, 
            model: str,
            prompt: Dict[str, str],
            stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Uses Langchain's Ollama wrapper to generate text.

        Args:
            model (str): The model to use.
            prompt (str): The input prompt.
            stream (bool): If True, stream responses.

        Yields:
            dict: JSON response from Langchain's Ollama model.
        """
        try:
            client = Ollama(model=model, base_url=self.base_url)

            prompt = prompt['system'] + "\n" + prompt['user']

            if stream:
                async for chunk in client.astream(prompt):
                    yield {"response": chunk}
            else:
                response = client.invoke(prompt)
                yield {"response": response}

        except Exception as e:
            raise RuntimeError(f"Error in OllamaService: {str(e)}") from e
