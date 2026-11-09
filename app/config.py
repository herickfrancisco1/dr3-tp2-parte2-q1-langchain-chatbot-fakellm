"""Configurações da aplicação (variáveis de ambiente com prefixo APP_ ou arquivo .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações da aplicação. Cada campo pode ser sobrescrito por uma variável de ambiente com prefixo APP_ (ex.:
    APP_DEVICE=0) ou pelo arquivo .env.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    llm_provider: str = "fake"
    """'fake' (padrão, sem custo) ou 'openai' (requer `pip install langchain-openai` e OPENAI_API_KEY)."""

    openai_model: str = "gpt-4o-mini"
    faq_path: Path = Path(__file__).parent / "data" / "faq.json"


@lru_cache
def get_settings() -> Settings:
    """Devolve as configurações. O @lru_cache faz o .env e as variáveis de ambiente serem lidos uma única vez."""
    return Settings()
