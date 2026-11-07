from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, examples=["O que é FastAPI?"])

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a mensagem não pode conter apenas espaços")
        return value.strip()


class ChatResponse(BaseModel):
    message: str
    answer: str
    llm: str
    elapsed_ms: float


class FaqItem(BaseModel):
    intent: str
    example: str
