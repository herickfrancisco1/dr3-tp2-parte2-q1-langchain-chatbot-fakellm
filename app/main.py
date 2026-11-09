"""Chatbot simples com FastAPI + LangChain + Fake LLM.

Execução:  uvicorn app.main:app --reload
Docs:      http://127.0.0.1:8000/docs
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from langchain_core.runnables import Runnable

from app.chain import build_chain, build_llm
from app.config import get_settings
from app.fake_llm import load_faq
from app.schemas import ChatRequest, ChatResponse, FaqItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Executado ao iniciar a API: cria o LLM (fake por padrão) e monta a cadeia uma única vez."""
    settings = get_settings()
    llm = build_llm(settings)
    app.state.chain = build_chain(llm)
    app.state.llm_name = llm._llm_type
    logger.info("Chatbot pronto (LLM: %s)", app.state.llm_name)
    yield


app = FastAPI(
    title="Fake LLM Chatbot",
    description="Protótipo de chatbot com LangChain usando um Fake LLM de respostas pré-definidas.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_chain(request: Request) -> Runnable:
    """Dependência do FastAPI: entrega à rota a cadeia montada no startup (ou HTTP 503)."""
    chain = getattr(request.app.state, "chain", None)
    if chain is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Chatbot ainda não inicializado.")
    return chain


@app.get("/health", tags=["infra"])
def health(request: Request) -> dict:
    """Rota de verificação (GET /health): informa se a API está no ar e qual LLM está em uso."""
    return {"status": "ok", "llm": getattr(request.app.state, "llm_name", None)}


@app.get("/faq", response_model=list[FaqItem], tags=["chat"])
def faq() -> list[FaqItem]:
    """Lista as perguntas pré-definidas que o Fake LLM sabe responder."""
    return [FaqItem(intent=e.intent, example=e.example) for e in load_faq(get_settings().faq_path)]


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest, request: Request, chain: Runnable = Depends(get_chain)) -> ChatResponse:
    """Rota POST /chat: envia a mensagem validada à cadeia LangChain (de forma assíncrona) e devolve a resposta em
    JSON.
    """
    start = time.perf_counter()
    answer = await chain.ainvoke({"question": payload.message})
    return ChatResponse(
        message=payload.message,
        answer=answer,
        llm=request.app.state.llm_name,
        elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
    )
