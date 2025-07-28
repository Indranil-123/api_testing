from pydantic import BaseModel



class TextInput(BaseModel):
    text : str


class IntentResponse(BaseModel):
    intent : str
    confidence : float
    response : str
    detected_language : str
    