"""Modelos Pydantic de entrada e saída da API (validação + documentação automática no /docs)."""

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Corpo da requisição POST /chat: a mensagem do usuário."""

    message: str = Field(..., min_length=1, max_length=500, examples=["O que é FastAPI?"])

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        """Rejeita mensagens só com espaços (HTTP 422) e remove os espaços das pontas."""
        if not value.strip():
            raise ValueError("a mensagem não pode conter apenas espaços")
        return value.strip()


class ChatResponse(BaseModel):
    """Corpo da resposta de /chat: mensagem, resposta, LLM usado e tempo gasto."""

    message: str
    answer: str
    llm: str
    elapsed_ms: float


class FaqItem(BaseModel):
    """Item listado em GET /faq: a intenção e um exemplo de pergunta."""

    intent: str
    example: str
