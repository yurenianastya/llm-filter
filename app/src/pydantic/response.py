from typing import Dict, Optional
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    message: str

class ProcessingResult(BaseModel):
    status: bool = False
    error: Optional[str] = None
    classification_result: Dict = {}
    semantic_result: Dict = {}
    is_recurrent: Optional[bool] = None

class ModelResponsePayload(BaseModel):
    preprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    postprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    llm_output: str = ""

class ModelResponse(BaseModel):
    user_message: str = ""
    results: ModelResponsePayload = Field(default_factory=ModelResponsePayload)
