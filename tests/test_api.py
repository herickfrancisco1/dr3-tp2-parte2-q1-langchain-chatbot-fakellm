"""Testes HTTP da API do chatbot (o Fake LLM é leve, então o lifespan real é executado)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # o Fake LLM é leve, então o lifespan real pode rodar nos testes
        yield c


def test_chat_returns_predefined_answer(client):
    response = client.post("/chat", json={"message": "O que é FastAPI?"})
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "O que é FastAPI?"
    assert body["answer"].startswith("FastAPI é um framework")
    assert body["llm"] == "predefined-answers-fake-llm"


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "   "}, {"message": "a" * 501}])
def test_chat_validates_input(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_faq_lists_questions(client):
    items = client.get("/faq").json()
    assert {"intent": "horario", "example": "Qual o horário de atendimento?"} in items


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "llm": "predefined-answers-fake-llm"}
