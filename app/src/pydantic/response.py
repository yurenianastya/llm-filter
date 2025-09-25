from typing import Dict, Optional
from pydantic import BaseModel, Field

class UserInput(BaseModel):
    message: str

class ProcessingResult(BaseModel):
    status: bool = False
    error: Optional[str] = None
    classification_result: Dict[str, float] = {}
    semantic_result: float = 0.0
    is_recurrent_result: bool = False
    anomaly_result: float = 0.0
    mixed_language_result: float = 0.0

class ModelResponsePayload(BaseModel): 
    preprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    postprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    llm_output: str = ""

class ModelResponse(BaseModel):
    user_message: str = ""
    results: ModelResponsePayload = Field(default_factory=ModelResponsePayload)
