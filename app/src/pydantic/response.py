from typing import Dict
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    message: str

class ProcessingResult(BaseModel):
    status: bool = False
    error: str = Field(default="", exclude=lambda v: v == "")
    classification_result: Dict = {}
    semantic_result: Dict = {}

class ModelResponsePayload(BaseModel):
    preprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    postprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    llm_output: str = ""

class ModelResponse(BaseModel):
    user_message: str = ""
    results: ModelResponsePayload = Field(default_factory=ModelResponsePayload)
