from pydantic import BaseModel, validator

CURSE_WORDS = {"some_bad_word"}

# super simple model
class UserInput(BaseModel):
    text: str

    @validator('text')
    def check_for_curse_words(cls, v):
        v_lower = v.lower()
        if any(curse_word in v_lower for curse_word in CURSE_WORDS):
            raise ValueError("Input contains inappropriate language.")
        
        return v