import pytest

from app.chain import build_chain
from app.config import get_settings
from app.fake_llm import DEFAULT_FALLBACK, PredefinedAnswersLLM, load_faq, normalize

FAQ = load_faq(get_settings().faq_path)
ANSWERS = {entry.intent: entry.answer for entry in FAQ}


@pytest.fixture
def chain():
    return build_chain(PredefinedAnswersLLM(faq=FAQ))


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Olá!", "saudacao"),
        ("Qual o HORÁRIO de atendimento?", "horario"),
        ("O que é FastAPI?", "fastapi"),
        ("Para que serve o LangChain?", "langchain"),
        ("Por que usar um Fake LLM?", "fake_llm"),
        ("O que você faz?", "ajuda"),
        ("Valeu, tchau", "despedida"),
    ],
)
def test_chain_answers_predefined_questions(chain, question, intent):
    assert chain.invoke({"question": question}) == ANSWERS[intent]


def test_unknown_question_returns_fallback(chain):
    assert chain.invoke({"question": "Qual a capital da Mongólia?"}) == DEFAULT_FALLBACK


def test_keywords_use_word_boundaries(chain):
    # 'oi' (saudação) não pode casar dentro de 'noite' nem de 'oito'; 'boa noite' sim.
    assert chain.invoke({"question": "Boa noite"}) == ANSWERS["saudacao"]
    assert chain.invoke({"question": "São oito horas"}) == DEFAULT_FALLBACK


def test_prompt_instructions_do_not_influence_answer(chain):
    # O prompt contém 'assistente' e 'português', mas só a pergunta é considerada.
    assert chain.invoke({"question": "xyz"}) == DEFAULT_FALLBACK


def test_normalize_removes_accents_and_case():
    assert normalize("Horário ÁÉÍ") == "horario aei"
