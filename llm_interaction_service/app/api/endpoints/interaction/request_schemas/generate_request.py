from pydantic import BaseModel
from fastapi import Form

class GenerateRequest(BaseModel):
    model: str = Form(...)
    prompt: str = Form(None)
    stream: bool =Form(False)