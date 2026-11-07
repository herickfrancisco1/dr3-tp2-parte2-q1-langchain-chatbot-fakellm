"""Montagem da cadeia LangChain (LCEL): prompt -> LLM -> parser."""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from app.config import Settings
from app.fake_llm import QUESTION_MARKER, PredefinedAnswersLLM, load_faq

PROMPT = PromptTemplate.from_template(
    "Você é um assistente virtual educado e objetivo. Responda em português.\n"
    f"{QUESTION_MARKER} {{question}}\n"
    "Resposta:"
)


def build_llm(settings: Settings) -> BaseLanguageModel:
    """Fábrica do modelo: trocar o Fake LLM por um LLM real muda SÓ esta função."""
    if settings.llm_provider == "fake":
        return PredefinedAnswersLLM(faq=load_faq(settings.faq_path))
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI  # dependência opcional

        return ChatOpenAI(model=settings.openai_model, temperature=0)
    raise ValueError(f"Provedor de LLM desconhecido: {settings.llm_provider!r}")


def build_chain(llm: BaseLanguageModel) -> Runnable:
    return PROMPT | llm | StrOutputParser()
