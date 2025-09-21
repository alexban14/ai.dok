from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, HTTPException, Header, Request
from typing import Optional
from starlette.status import HTTP_401_UNAUTHORIZED
from app.core.config import config

def setup_cors(app: FastAPI):
    origins = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:3010",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8221",
        "http://localhost:8762",
        "https://docintellect.banalexandru.online",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

def authorize_client(
        client_id: int,
        authorization: Optional[str] = Header(None)
):
    if authorization is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header"
        )

    try:
        index = config.client_ids.index(client_id)
    except ValueError:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid client_id"
        )

    expected_token = config.api_access_tokens[index]

    if authorization != expected_token:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid token for given client_id"
        )

    return True