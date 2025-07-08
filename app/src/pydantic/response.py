from pydantic import BaseModel, Field


class UserInput(BaseModel):
    message: str

class FilterOutput(BaseModel):
    label: str = ""
    score: float = 0.0

class SemanticOutput(BaseModel):
    score: float = 0.0

class ProcessingResult(BaseModel):
    status: bool = False
    classification_result: FilterOutput = Field(default_factory=FilterOutput)
    semantic_result: SemanticOutput = Field(default_factory=SemanticOutput)

class ModelResponsePayload(BaseModel):
    preprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    postprocessing_result: ProcessingResult = Field(default_factory=ProcessingResult)
    llm_output: str = ""

class ModelResponse(BaseModel):
    user_message: str = ""
    results: ModelResponsePayload = Field(default_factory=ModelResponsePayload)