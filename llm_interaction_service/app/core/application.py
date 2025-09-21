from fastapi import FastAPI
from app.core.app_logger import setup_logging
from app.core.middleware import setup_cors
import logging
from app.api.endpoints import hello
from app.api.endpoints.interaction import process_file
from app.api.endpoints import indexing


def create_api():
    api = FastAPI()

    # Apply middleware
    setup_cors(api)

    setup_logging()
    logging.info("logging works!")

    api.include_router(hello.router)

    # llm interaction endpoints
    llmInteractionPrefix = '/llm-interaction-api/v1'
    api.include_router(process_file.router, prefix=llmInteractionPrefix, tags=['LLmInteractionApi', 'LlmProcessFile'])
    api.include_router(indexing.router, prefix='/indexing', tags=['Indexing'])

    return api