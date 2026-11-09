"""Fake LLM que responde perguntas pré-definidas (FAQ), implementado como um LLM do LangChain.

Por herdar de `langchain_core.language_models.llms.LLM`, ele se comporta como qualquer modelo real
(invoke, batch, stream, callbacks...), então a cadeia não precisa mudar quando ele for trocado por
um LLM de verdade.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel

# O prompt da cadeia coloca a pergunta do usuário depois deste marcador; o Fake LLM "lê" só essa parte
# (assim as palavras das instruções do prompt não interferem na escolha da resposta).
QUESTION_MARKER = "Pergunta:"

DEFAULT_FALLBACK = (
    "Desculpe, ainda não sei responder isso. Tente perguntar sobre horário, FastAPI, LangChain ou Fake LLM."
)


class FaqEntry(BaseModel):
    """Uma entrada da FAQ (faq.json): intenção, palavras-chave, exemplo de pergunta e resposta."""

    intent: str
    keywords: list[str]
    example: str
    answer: str


def normalize(text: str) -> str:
    """Minúsculas e sem acentos: 'Horário' -> 'horario'."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def load_faq(path: Path) -> list[FaqEntry]:
    """Lê o faq.json e valida cada item com o modelo FaqEntry."""
    return [FaqEntry(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


class PredefinedAnswersLLM(LLM):
    """Escolhe a resposta da FAQ com mais palavras-chave presentes na pergunta."""

    faq: list[FaqEntry]
    fallback: str = DEFAULT_FALLBACK

    @property
    def _llm_type(self) -> str:
        """Nome do tipo de LLM. É exigido pelo LangChain e aparece em logs e na resposta da API."""
        return "predefined-answers-fake-llm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Parâmetros que identificam este LLM (o LangChain os usa, por exemplo, no cache)."""
        return {"intents": [entry.intent for entry in self.faq]}

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Coração do Fake LLM: recebe o prompt pronto e devolve a resposta da FAQ com mais palavras-chave encontradas
        na pergunta (ou a resposta padrão).
        """
        question = normalize(prompt.rsplit(QUESTION_MARKER, 1)[-1])
        best_score, best_answer = 0, self.fallback
        for entry in self.faq:
            # \b evita falsos positivos (ex.: 'oi' dentro de 'noite').
            score = sum(1 for kw in entry.keywords if re.search(rf"\b{re.escape(normalize(kw))}\b", question))
            if score > best_score:
                best_score, best_answer = score, entry.answer
        return best_answer
