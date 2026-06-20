from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/resume_scoring"

    # LLM
    LLM_PROVIDER: Literal["deepseek", "openrouter", "anthropic"] = "deepseek"
    LLM_MODEL: str = "deepseek-chat"
    DEEPSEEK_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # HH.ru
    HH_CLIENT_ID: str = ""
    HH_CLIENT_SECRET: str = ""
    HH_REDIRECT_URI: str = "http://localhost:8000/auth/hh/callback"
    HH_EMPLOYER_ID: str = ""
    HH_ACCESS_TOKEN: str = ""   # сохраняется после OAuth2

    # SuperJob
    SUPERJOB_API_KEY: str = ""
    SUPERJOB_CLIENT_ID: str = ""
    SUPERJOB_EMPLOYER_ID: str = ""

    # Scheduler
    SYNC_INTERVAL_HOURS: int = 6

    # App
    DEBUG: bool = False


settings = Settings()
