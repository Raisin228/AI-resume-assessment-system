# Scoring Module — Plan реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать Scoring Module — автоматическое ранжирование кандидатов по вакансии через LLM, с синхронизацией из HH.ru и SuperJob, LangGraph workflow-пайплайном и веб-интерфейсом для рекрутёра.

**Architecture:** Три LLM-вызова (Parse Vacancy → Parse Resume → Score) реализованы как LangGraph StateGraph с фиксированными рёбрами (не автономный агент — LLM не решает что вызывать, данные всегда подаются из Python-кода). Батч-обработка: `asyncio.gather + Semaphore(5)`. Два источника данных: HH.ru (OAuth2, полная автоматизация) + SuperJob (API-ключ). Scheduler: APScheduler каждые 6ч. Web UI: FastAPI + Jinja2 + HTMX + SSE-прогресс.

**Tech Stack:** Python 3.12, Poetry, FastAPI, SQLAlchemy 2.0 (async), Alembic, PostgreSQL, APScheduler, LangGraph, langchain-openai, httpx, pdfminer.six, pydantic-settings, Jinja2, HTMX, sse-starlette, Docker Compose

## Global Constraints

- Python 3.12, Poetry — не pip, не uv
- Конфиг только через `pydantic-settings` + `.env` — никакого `os.environ` напрямую
- LLM output: function calling → Pydantic (не парсинг текста)
- `temperature=0` для всех LLM-вызовов
- Резюме и JD не логируются в stdout (персональные данные)
- `LLM_PROVIDER=deepseek|openrouter|anthropic` (дефолт: deepseek)
- UI: Jinja2 + HTMX — не React, не Streamlit
- Async везде где есть I/O
- Type hints обязательны везде
- Semaphore(5) для батч-обработки

---

## Файловая структура

```
app/
  core/
    config.py           # pydantic-settings Settings
    database.py         # async SQLAlchemy engine + session factory
  models/
    base.py             # DeclarativeBase
    vacancy.py          # Vacancy ORM model
    candidate.py        # Candidate ORM model
    scoring_result.py   # ScoringResult ORM model
    sync_log.py         # SyncLog ORM model
  schemas/
    vacancy.py          # ParsedVacancy + PARSE_VACANCY_TOOL_SCHEMA
    candidate.py        # ParsedResume + PARSE_RESUME_TOOL_SCHEMA
    scoring.py          # CandidateScore + CANDIDATE_SCORE_TOOL_SCHEMA
  llm/
    client.py           # LLMClient Protocol
    deepseek.py         # DeepSeekClient (openai-compatible)
    openrouter.py       # OpenRouterClient (openai-compatible)
    anthropic_client.py # AnthropicClient (native SDK)
    factory.py          # get_llm_client() → LLMClient
  pipeline/
    prompts.py          # PARSE_VACANCY_SYSTEM, PARSE_RESUME_SYSTEM, SCORE_SYSTEM
    parse_vacancy.py    # parse_vacancy(vacancy_id) → ParsedVacancy
    parse_resume.py     # parse_resume(candidate_id) → ParsedResume
    scorer.py           # score_pair(vacancy_id, candidate_id, ...) → CandidateScore
    graph.py            # LangGraph StateGraph (ScoringState, build_scoring_graph)
    orchestrator.py     # score_candidates_batch() — asyncio + Semaphore
  integrations/
    hh/
      client.py         # HHClient — OAuth2 + API calls
      sync_worker.py    # sync_hh_vacancies_and_candidates()
      mapper.py         # hh_resume_json → raw_text string
    superjob/
      client.py         # SuperJobClient — API key auth
      sync_worker.py    # sync_superjob_vacancies_and_candidates()
      mapper.py         # superjob_resume_json → raw_text string
  scheduler.py          # APScheduler setup + job registration
  web/
    routes/
      vacancies.py      # GET /vacancies, GET /vacancies/{id}
      candidates.py     # GET /vacancies/{id}/candidates, GET /candidates/{id}
      upload.py         # GET/POST /upload (ручной режим)
      sse.py            # GET /score/progress/{job_id} — SSE stream
      api.py            # POST /api/score — trigger batch scoring
    templates/
      base.html
      vacancies/
        list.html       # список вакансий компании
        detail.html     # ранжированная таблица кандидатов по вакансии
      candidates/
        detail.html     # карточка кандидата — оценка, матчи, пробелы
      upload.html       # форма ручной загрузки
  main.py               # FastAPI app + lifespan + router registration
alembic/
  versions/
    001_initial_schema.py
  env.py
alembic.ini
tests/
  conftest.py
  test_llm/
    test_client.py
    test_parsers.py
    test_scorer.py
  test_integrations/
    test_hh_client.py
    test_superjob_client.py
  test_pipeline/
    test_graph.py
    test_orchestrator.py
  test_web/
    test_vacancies.py
    test_upload.py
pyproject.toml
docker-compose.yml
Dockerfile
.env.example
```

---

### Task 1: Infrastructure & Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Create: `.env.example`
- Create: `app/core/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `settings` singleton используется всеми модулями

- [ ] **Step 1: Инициализировать Poetry-проект**

```bash
poetry init --name ai-resume-assessment --python "^3.12" --no-interaction
poetry add fastapi "uvicorn[standard]" "sqlalchemy[asyncio]" asyncpg alembic \
  "pydantic>=2.7" pydantic-settings httpx \
  langgraph langchain-openai langchain-anthropic \
  "apscheduler>=3.10" jinja2 python-multipart pdfminer-six sse-starlette \
  openai anthropic
poetry add --group dev pytest pytest-asyncio httpx
```

- [ ] **Step 2: Написать `app/core/config.py`**

```python
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
```

- [ ] **Step 3: Написать `.env.example`**

```ini
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/resume_scoring
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your_deepseek_key_here
HH_CLIENT_ID=
HH_CLIENT_SECRET=
HH_EMPLOYER_ID=
HH_ACCESS_TOKEN=
SUPERJOB_API_KEY=
SUPERJOB_CLIENT_ID=
SUPERJOB_EMPLOYER_ID=
SYNC_INTERVAL_HOURS=6
DEBUG=False
```

- [ ] **Step 4: Написать `docker-compose.yml`**

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: resume_scoring
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      db:
        condition: service_healthy
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  pgdata:
```

- [ ] **Step 5: Написать `Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install poetry==1.8.3
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --no-root
COPY . .
```

- [ ] **Step 6: Написать failing тест**

```python
# tests/test_config.py
from app.core.config import Settings

def test_settings_load_defaults():
    s = Settings()
    assert s.LLM_PROVIDER == "deepseek"
    assert s.LLM_MODEL == "deepseek-chat"
    assert s.SYNC_INTERVAL_HOURS == 6

def test_settings_override_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    s = Settings()
    assert s.LLM_PROVIDER == "openrouter"
    assert s.LLM_MODEL == "gpt-4o-mini"
```

- [ ] **Step 7: Запустить тест**

```bash
pytest tests/test_config.py -v
# Expected: 2 passed
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml docker-compose.yml Dockerfile .env.example app/core/config.py tests/test_config.py
git commit -m "feat: project infrastructure, docker-compose, pydantic-settings config"
```

---

### Task 2: Database Schema & Migrations

**Files:**
- Create: `app/models/base.py`
- Create: `app/models/vacancy.py`
- Create: `app/models/candidate.py`
- Create: `app/models/scoring_result.py`
- Create: `app/models/sync_log.py`
- Create: `app/core/database.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/001_initial_schema.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `get_session()` async context manager используется всеми pipeline- и web-модулями
- Produces: `Vacancy`, `Candidate`, `ScoringResult`, `SyncLog` ORM-классы

- [ ] **Step 1: Написать `app/models/base.py`**

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

- [ ] **Step 2: Написать `app/models/vacancy.py`**

```python
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class Vacancy(Base):
    __tablename__ = "vacancies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hh_vacancy_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    superjob_vacancy_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    parsed: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    employer_name: Mapped[str | None] = mapped_column(String, nullable=True)
    salary_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String(10), nullable=True)
    area: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # hh | superjob | manual
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

- [ ] **Step 3: Написать `app/models/candidate.py`**

```python
import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hh_resume_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    superjob_resume_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    parsed: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # hh | superjob | manual
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
```

- [ ] **Step 4: Написать `app/models/scoring_result.py`**

```python
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, Float, DateTime, ForeignKey, func, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class ScoringResult(Base):
    __tablename__ = "scoring_results"
    __table_args__ = (
        UniqueConstraint("vacancy_id", "candidate_id", name="uq_scoring_vacancy_candidate"),
        CheckConstraint("score >= 0 AND score <= 100", name="ck_score_range"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vacancy_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("vacancies.id"), nullable=False)
    candidate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    grade: Mapped[str] = mapped_column(String(20), nullable=False)  # strong_match | good_match | weak_match | no_match
    key_matches: Mapped[list] = mapped_column(JSONB, nullable=False)
    key_gaps: Mapped[list] = mapped_column(JSONB, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_used: Mapped[str] = mapped_column(String(60), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
```

- [ ] **Step 5: Написать `app/models/sync_log.py`**

```python
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base


class SyncLog(Base):
    __tablename__ = "sync_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # hh | superjob
    vacancy_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("vacancies.id"), nullable=True)
    candidates_fetched: Mapped[int] = mapped_column(Integer, default=0)
    candidates_new: Mapped[int] = mapped_column(Integer, default=0)
    candidates_scored: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # success | partial | failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 6: Написать `app/core/database.py`**

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

- [ ] **Step 7: Инициализировать Alembic**

```bash
alembic init alembic
```

- [ ] **Step 8: Отредактировать `alembic/env.py`**

Заменить содержимое на:

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from app.core.config import settings
from app.models.base import Base
from app.models.vacancy import Vacancy       # noqa: F401
from app.models.candidate import Candidate   # noqa: F401
from app.models.scoring_result import ScoringResult  # noqa: F401
from app.models.sync_log import SyncLog      # noqa: F401

config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("+asyncpg", ""))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 9: Сгенерировать миграцию и применить**

```bash
docker compose up db -d
alembic revision --autogenerate -m "initial_schema"
alembic upgrade head
# Expected: 4 таблицы созданы в БД
```

- [ ] **Step 10: Написать тест**

```python
# tests/test_models.py
import pytest
import asyncio
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.models.base import Base
from app.models.vacancy import Vacancy
from app.models.candidate import Candidate
from app.models.scoring_result import ScoringResult

TEST_DB = "postgresql+asyncpg://postgres:postgres@localhost:5432/resume_scoring"

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def db_session():
    engine = create_async_engine(TEST_DB)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.mark.asyncio
async def test_vacancy_create(db_session):
    v = Vacancy(title="Python Dev", description="FastAPI required", source="manual")
    db_session.add(v)
    await db_session.commit()
    assert v.id is not None
    assert v.status == "active"

@pytest.mark.asyncio
async def test_scoring_result_unique_constraint(db_session):
    v = Vacancy(title="Go Dev", description="Go required", source="hh")
    c = Candidate(raw_text="Go developer 3 years", source="hh")
    db_session.add_all([v, c])
    await db_session.commit()

    r1 = ScoringResult(
        vacancy_id=v.id, candidate_id=c.id, score=75, grade="good_match",
        key_matches=["Go"], key_gaps=[], summary="Good fit", confidence=0.9,
        model_used="deepseek-chat",
    )
    db_session.add(r1)
    await db_session.commit()

    from sqlalchemy.exc import IntegrityError
    r2 = ScoringResult(
        vacancy_id=v.id, candidate_id=c.id, score=80, grade="strong_match",
        key_matches=["Go"], key_gaps=[], summary="Better fit", confidence=0.95,
        model_used="deepseek-chat",
    )
    db_session.add(r2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
```

- [ ] **Step 11: Запустить тест**

```bash
pytest tests/test_models.py -v
# Expected: 2 passed
```

- [ ] **Step 12: Commit**

```bash
git add app/models/ app/core/database.py alembic/ alembic.ini tests/test_models.py
git commit -m "feat: SQLAlchemy models (4 tables), Alembic migration, async session factory"
```

---

### Task 3: Abstract LLM Client Layer

**Files:**
- Create: `app/llm/client.py`
- Create: `app/llm/deepseek.py`
- Create: `app/llm/openrouter.py`
- Create: `app/llm/anthropic_client.py`
- Create: `app/llm/factory.py`
- Test: `tests/test_llm/test_client.py`

**Interfaces:**
- Produces: `get_llm_client() → LLMClient` — используется в `parse_vacancy`, `parse_resume`, `scorer`
- Produces: `LLMClient.structured_call(system, user, tool_schema) → dict`

- [ ] **Step 1: Написать `app/llm/client.py`** (Protocol)

```python
from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    async def structured_call(
        self,
        system: str,
        user: str,
        tool_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Вызов LLM с function calling. Возвращает распарсенный dict аргументов."""
        ...
```

- [ ] **Step 2: Написать `app/llm/deepseek.py`**

```python
import json
from typing import Any
from openai import AsyncOpenAI
from app.core.config import settings


class DeepSeekClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
        self._model = settings.LLM_MODEL or "deepseek-chat"

    async def structured_call(
        self,
        system: str,
        user: str,
        tool_schema: dict[str, Any],
    ) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[{"type": "function", "function": tool_schema}],
            tool_choice={"type": "function", "function": {"name": tool_schema["name"]}},
            temperature=0,
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
```

- [ ] **Step 3: Написать `app/llm/openrouter.py`**

```python
import json
from typing import Any
from openai import AsyncOpenAI
from app.core.config import settings


class OpenRouterClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        self._model = settings.LLM_MODEL or "anthropic/claude-haiku-4-5"

    async def structured_call(
        self,
        system: str,
        user: str,
        tool_schema: dict[str, Any],
    ) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[{"type": "function", "function": tool_schema}],
            tool_choice={"type": "function", "function": {"name": tool_schema["name"]}},
            temperature=0,
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
```

- [ ] **Step 4: Написать `app/llm/anthropic_client.py`**

```python
import json
from typing import Any
import anthropic
from app.core.config import settings


class AnthropicClient:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self._model = settings.LLM_MODEL or "claude-haiku-4-5-20251001"

    async def structured_call(
        self,
        system: str,
        user: str,
        tool_schema: dict[str, Any],
    ) -> dict[str, Any]:
        tool = {
            "name": tool_schema["name"],
            "description": tool_schema.get("description", ""),
            "input_schema": tool_schema["parameters"],
        }
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        raise ValueError("Anthropic response contained no tool_use block")
```

- [ ] **Step 5: Написать `app/llm/factory.py`**

```python
from app.llm.client import LLMClient
from app.core.config import settings


def get_llm_client() -> LLMClient:
    match settings.LLM_PROVIDER:
        case "deepseek":
            from app.llm.deepseek import DeepSeekClient
            return DeepSeekClient()
        case "openrouter":
            from app.llm.openrouter import OpenRouterClient
            return OpenRouterClient()
        case "anthropic":
            from app.llm.anthropic_client import AnthropicClient
            return AnthropicClient()
        case _:
            raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")
```

- [ ] **Step 6: Написать тест (мокируем HTTP)**

```python
# tests/test_llm/test_client.py
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from app.llm.deepseek import DeepSeekClient
from app.llm.factory import get_llm_client

TOOL_SCHEMA = {
    "name": "test_tool",
    "description": "Test",
    "parameters": {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    },
}

@pytest.mark.asyncio
async def test_deepseek_client_structured_call():
    client = DeepSeekClient()

    tool_call_mock = MagicMock()
    tool_call_mock.function.arguments = json.dumps({"result": "hello"})

    choice_mock = MagicMock()
    choice_mock.message.tool_calls = [tool_call_mock]

    response_mock = MagicMock()
    response_mock.choices = [choice_mock]

    with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=response_mock)):
        result = await client.structured_call(
            system="You are a test assistant",
            user="Return hello",
            tool_schema=TOOL_SCHEMA,
        )

    assert result == {"result": "hello"}

def test_factory_returns_deepseek(monkeypatch):
    monkeypatch.setattr("app.llm.factory.settings.LLM_PROVIDER", "deepseek")
    from app.llm.deepseek import DeepSeekClient
    client = get_llm_client()
    assert isinstance(client, DeepSeekClient)

def test_factory_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr("app.llm.factory.settings.LLM_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm_client()
```

- [ ] **Step 7: Запустить тест**

```bash
pytest tests/test_llm/test_client.py -v
# Expected: 3 passed
```

- [ ] **Step 8: Commit**

```bash
git add app/llm/ tests/test_llm/test_client.py
git commit -m "feat: abstract LLMClient Protocol with DeepSeek, OpenRouter, Anthropic backends"
```

---

### Task 4: Pydantic Schemas + Tool Schemas

**Files:**
- Create: `app/schemas/vacancy.py`
- Create: `app/schemas/candidate.py`
- Create: `app/schemas/scoring.py`
- Test: `tests/test_llm/test_schemas.py`

**Interfaces:**
- Produces: `ParsedVacancy`, `PARSE_VACANCY_TOOL_SCHEMA` — используется в `parse_vacancy.py`
- Produces: `ParsedResume`, `PARSE_RESUME_TOOL_SCHEMA` — используется в `parse_resume.py`
- Produces: `CandidateScore`, `CANDIDATE_SCORE_TOOL_SCHEMA` — используется в `scorer.py`

- [ ] **Step 1: Написать `app/schemas/vacancy.py`**

```python
from pydantic import BaseModel

class ParsedVacancy(BaseModel):
    required_skills: list[str]
    nice_to_have: list[str]
    min_experience_years: int
    level: str  # junior | middle | senior | lead
    responsibilities: list[str]

PARSE_VACANCY_TOOL_SCHEMA: dict = {
    "name": "extract_vacancy_info",
    "description": "Extract structured requirements from a job vacancy description",
    "parameters": {
        "type": "object",
        "properties": {
            "required_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required technical skills and technologies",
            },
            "nice_to_have": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional or nice-to-have skills",
            },
            "min_experience_years": {
                "type": "integer",
                "description": "Minimum years of experience required. Use 0 if not specified.",
            },
            "level": {
                "type": "string",
                "enum": ["junior", "middle", "senior", "lead"],
                "description": "Seniority level required",
            },
            "responsibilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key job responsibilities, top 5 max",
            },
        },
        "required": ["required_skills", "nice_to_have", "min_experience_years", "level", "responsibilities"],
    },
}
```

- [ ] **Step 2: Написать `app/schemas/candidate.py`**

```python
from pydantic import BaseModel

class ParsedResume(BaseModel):
    skills: list[str]
    experience_years: int
    education: str
    last_position: str
    summary: str

PARSE_RESUME_TOOL_SCHEMA: dict = {
    "name": "extract_resume_info",
    "description": "Extract structured information from a candidate resume",
    "parameters": {
        "type": "object",
        "properties": {
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Technical skills and technologies the candidate has",
            },
            "experience_years": {
                "type": "integer",
                "description": "Total years of relevant professional work experience",
            },
            "education": {
                "type": "string",
                "description": "Highest degree and field, e.g. 'МГТУ, Прикладная математика, бакалавр'",
            },
            "last_position": {
                "type": "string",
                "description": "Most recent job title and company, e.g. 'Backend Engineer @ Tinkoff'",
            },
            "summary": {
                "type": "string",
                "description": "Brief 1-2 sentence professional summary",
            },
        },
        "required": ["skills", "experience_years", "education", "last_position", "summary"],
    },
}
```

- [ ] **Step 3: Написать `app/schemas/scoring.py`**

```python
from typing import Literal
from pydantic import BaseModel, Field

class CandidateScore(BaseModel):
    score: int = Field(ge=0, le=100)
    grade: Literal["strong_match", "good_match", "weak_match", "no_match"]
    key_matches: list[str] = Field(max_length=3)
    key_gaps: list[str] = Field(max_length=3)
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)

CANDIDATE_SCORE_TOOL_SCHEMA: dict = {
    "name": "evaluate_candidate_fit",
    "description": "Score a candidate against a job vacancy requirements",
    "parameters": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Overall fit score 0-100",
            },
            "grade": {
                "type": "string",
                "enum": ["strong_match", "good_match", "weak_match", "no_match"],
            },
            "key_matches": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": "Top 3 reasons why this candidate is suitable",
            },
            "key_gaps": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": "Top 3 gaps or concerns. MUST be empty [] if candidate meets ALL requirements.",
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence assessment in Russian, addressed to the recruiter",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in this assessment: 0.9 if data complete, 0.7 if sparse",
            },
        },
        "required": ["score", "grade", "key_matches", "key_gaps", "summary", "confidence"],
    },
}
```

- [ ] **Step 4: Написать тест**

```python
# tests/test_llm/test_schemas.py
from pydantic import ValidationError
import pytest
from app.schemas.vacancy import ParsedVacancy
from app.schemas.candidate import ParsedResume
from app.schemas.scoring import CandidateScore

def test_parsed_vacancy_valid():
    v = ParsedVacancy(
        required_skills=["Python", "FastAPI"],
        nice_to_have=["Kubernetes"],
        min_experience_years=3,
        level="middle",
        responsibilities=["Разработка API", "Code review"],
    )
    assert v.level == "middle"
    assert len(v.required_skills) == 2

def test_candidate_score_key_gaps_empty_for_perfect():
    s = CandidateScore(
        score=95, grade="strong_match", key_matches=["Python", "FastAPI", "3+ лет"],
        key_gaps=[], summary="Отличный кандидат.", confidence=0.95,
    )
    assert s.key_gaps == []

def test_candidate_score_invalid_score():
    with pytest.raises(ValidationError):
        CandidateScore(
            score=150, grade="strong_match", key_matches=[], key_gaps=[],
            summary="Test", confidence=0.9,
        )
```

- [ ] **Step 5: Запустить тест**

```bash
pytest tests/test_llm/test_schemas.py -v
# Expected: 3 passed
```

- [ ] **Step 6: Commit**

```bash
git add app/schemas/ tests/test_llm/test_schemas.py
git commit -m "feat: Pydantic schemas and tool schemas for vacancy, resume, scoring"
```

---

### Task 5: Prompts & LLM Pipeline Steps (Parse + Score)

**Files:**
- Create: `app/pipeline/prompts.py`
- Create: `app/pipeline/parse_vacancy.py`
- Create: `app/pipeline/parse_resume.py`
- Create: `app/pipeline/scorer.py`
- Test: `tests/test_llm/test_parsers.py`

**Interfaces:**
- Consumes: `get_llm_client()`, `get_session()`, `Vacancy`, `Candidate`, `ScoringResult`
- Consumes: `ParsedVacancy`, `ParsedResume`, `CandidateScore` + их tool schemas
- Produces: `parse_vacancy(vacancy_id) → ParsedVacancy`
- Produces: `parse_resume(candidate_id) → ParsedResume`
- Produces: `score_pair(vacancy_id, candidate_id, parsed_vacancy, parsed_resume) → CandidateScore`

- [ ] **Step 1: Написать `app/pipeline/prompts.py`**

```python
PARSE_VACANCY_SYSTEM = """You are a recruitment data extractor.
Extract structured information from job vacancy descriptions.
Extract ONLY what is explicitly stated. For missing fields use empty lists or 0.
Do not infer or hallucinate requirements."""

PARSE_RESUME_SYSTEM = """You are a resume data extractor.
Extract structured information from candidate resumes.
Extract ONLY what is explicitly mentioned. Do not infer or hallucinate skills or experience.
For experience_years: sum all relevant positions, round to nearest integer."""

SCORE_SYSTEM = """You are an expert technical recruiter evaluating candidate-vacancy fit for Russian IT companies.

## Scoring Framework

Evaluate the following parameters and combine into a final score 0-100:

1. **Technical Skills Match** (40% weight)
   - All required_skills present in candidate.skills → +40
   - Each missing required skill: -8 (deduct from 40)
   - Each nice_to_have present: +3 (max +12 total)

2. **Experience Level** (25% weight)
   - candidate.experience_years >= min_experience_years → +25
   - Deficit of 1 year: -10; deficit of 2+ years: -20

3. **Seniority Alignment** (20% weight)
   - Required level matches or candidate is overqualified by 1 → +20
   - Underqualified by 1 level: -15; by 2+ levels: -25

4. **Domain Relevance** (15% weight)
   - Same or adjacent industry/domain: +15
   - Completely unrelated domain: 0

## Grade thresholds
- 80-100: strong_match
- 60-79: good_match
- 40-59: weak_match
- 0-39: no_match

## Critical rules
- key_gaps MUST be [] (empty array) if candidate meets ALL required criteria without any gap
- key_matches: exactly top-3 strongest reasons for fit (fewer only if fewer exist)
- key_gaps: top-3 most significant gaps ([] if none)
- summary: 2-3 sentences in Russian, addressed to the recruiter, professional tone
- confidence: 0.9 if resume is detailed; 0.7 if resume is sparse; 0.5 if critical fields missing"""
```

- [ ] **Step 2: Написать `app/pipeline/parse_vacancy.py`**

```python
import uuid
from sqlalchemy import select
from app.core.database import get_session
from app.models.vacancy import Vacancy
from app.schemas.vacancy import ParsedVacancy, PARSE_VACANCY_TOOL_SCHEMA
from app.pipeline.prompts import PARSE_VACANCY_SYSTEM
from app.llm.factory import get_llm_client


async def parse_vacancy(vacancy_id: uuid.UUID) -> ParsedVacancy:
    """Idempotent: если vacancy.parsed уже заполнен — возвращает кэш, LLM не вызывается."""
    async with get_session() as session:
        result = await session.execute(select(Vacancy).where(Vacancy.id == vacancy_id))
        vacancy = result.scalar_one()

        if vacancy.parsed is not None:
            return ParsedVacancy.model_validate(vacancy.parsed)

        client = get_llm_client()
        raw = await client.structured_call(
            system=PARSE_VACANCY_SYSTEM,
            user=vacancy.description,
            tool_schema=PARSE_VACANCY_TOOL_SCHEMA,
        )
        parsed = ParsedVacancy.model_validate(raw)
        vacancy.parsed = parsed.model_dump()
        await session.commit()
        return parsed
```

- [ ] **Step 3: Написать `app/pipeline/parse_resume.py`**

```python
import uuid
from sqlalchemy import select
from app.core.database import get_session
from app.models.candidate import Candidate
from app.schemas.candidate import ParsedResume, PARSE_RESUME_TOOL_SCHEMA
from app.pipeline.prompts import PARSE_RESUME_SYSTEM
from app.llm.factory import get_llm_client


async def parse_resume(candidate_id: uuid.UUID) -> ParsedResume:
    """Idempotent: если candidate.parsed уже заполнен — возвращает кэш."""
    async with get_session() as session:
        result = await session.execute(select(Candidate).where(Candidate.id == candidate_id))
        candidate = result.scalar_one()

        if candidate.parsed is not None:
            return ParsedResume.model_validate(candidate.parsed)

        client = get_llm_client()
        raw = await client.structured_call(
            system=PARSE_RESUME_SYSTEM,
            user=candidate.raw_text,
            tool_schema=PARSE_RESUME_TOOL_SCHEMA,
        )
        parsed = ParsedResume.model_validate(raw)
        candidate.parsed = parsed.model_dump()
        await session.commit()
        return parsed
```

- [ ] **Step 4: Написать `app/pipeline/scorer.py`**

```python
import uuid
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.core.database import get_session
from app.core.config import settings
from app.models.scoring_result import ScoringResult
from app.schemas.vacancy import ParsedVacancy
from app.schemas.candidate import ParsedResume
from app.schemas.scoring import CandidateScore, CANDIDATE_SCORE_TOOL_SCHEMA
from app.pipeline.prompts import SCORE_SYSTEM
from app.llm.factory import get_llm_client


def _build_score_prompt(vacancy: ParsedVacancy, resume: ParsedResume) -> str:
    return f"""## Vacancy Requirements
Level: {vacancy.level}
Min Experience: {vacancy.min_experience_years} years
Required Skills: {", ".join(vacancy.required_skills)}
Nice to Have: {", ".join(vacancy.nice_to_have)}
Responsibilities: {"; ".join(vacancy.responsibilities[:3])}

## Candidate Profile
Skills: {", ".join(resume.skills)}
Experience: {resume.experience_years} years
Last Position: {resume.last_position}
Education: {resume.education}
Summary: {resume.summary}

Evaluate this candidate against the vacancy requirements."""


async def score_pair(
    vacancy_id: uuid.UUID,
    candidate_id: uuid.UUID,
    parsed_vacancy: ParsedVacancy,
    parsed_resume: ParsedResume,
) -> CandidateScore:
    """Idempotent: если результат для пары уже существует — возвращает кэш."""
    async with get_session() as session:
        existing = await session.execute(
            select(ScoringResult).where(
                ScoringResult.vacancy_id == vacancy_id,
                ScoringResult.candidate_id == candidate_id,
            )
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            return CandidateScore(
                score=row.score, grade=row.grade,
                key_matches=row.key_matches, key_gaps=row.key_gaps,
                summary=row.summary, confidence=row.confidence,
            )

        client = get_llm_client()
        raw = await client.structured_call(
            system=SCORE_SYSTEM,
            user=_build_score_prompt(parsed_vacancy, parsed_resume),
            tool_schema=CANDIDATE_SCORE_TOOL_SCHEMA,
        )
        result = CandidateScore.model_validate(raw)

        db_result = ScoringResult(
            vacancy_id=vacancy_id,
            candidate_id=candidate_id,
            score=result.score,
            grade=result.grade,
            key_matches=result.key_matches,
            key_gaps=result.key_gaps,
            summary=result.summary,
            confidence=result.confidence,
            model_used=settings.LLM_MODEL,
        )
        session.add(db_result)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
        return result
```

- [ ] **Step 5: Написать тест (мокируем LLM)**

```python
# tests/test_llm/test_parsers.py
import pytest
import uuid
from unittest.mock import AsyncMock, patch
from app.schemas.vacancy import ParsedVacancy
from app.schemas.candidate import ParsedResume
from app.schemas.scoring import CandidateScore
from app.pipeline.scorer import _build_score_prompt

def test_build_score_prompt_contains_skills():
    v = ParsedVacancy(
        required_skills=["Python", "FastAPI"],
        nice_to_have=["Redis"],
        min_experience_years=3,
        level="middle",
        responsibilities=["Разработка API"],
    )
    r = ParsedResume(
        skills=["Python", "FastAPI", "PostgreSQL"],
        experience_years=4,
        education="МГТУ, CS",
        last_position="Backend @ Yandex",
        summary="Python developer",
    )
    prompt = _build_score_prompt(v, r)
    assert "Python" in prompt
    assert "FastAPI" in prompt
    assert "4 years" in prompt

def test_candidate_score_perfect_no_gaps():
    s = CandidateScore(
        score=92,
        grade="strong_match",
        key_matches=["Python", "FastAPI", "4 лет опыта"],
        key_gaps=[],
        summary="Отличный кандидат.",
        confidence=0.93,
    )
    assert s.key_gaps == []
    assert s.grade == "strong_match"
```

- [ ] **Step 6: Запустить тест**

```bash
pytest tests/test_llm/test_parsers.py -v
# Expected: 2 passed
```

- [ ] **Step 7: Commit**

```bash
git add app/pipeline/ tests/test_llm/test_parsers.py
git commit -m "feat: LLM pipeline steps — parse_vacancy, parse_resume, scorer with idempotent caching"
```

---

### Task 6: LangGraph Scoring Workflow + Batch Orchestrator

**Files:**
- Create: `app/pipeline/graph.py`
- Create: `app/pipeline/orchestrator.py`
- Test: `tests/test_pipeline/test_graph.py`

**Interfaces:**
- Consumes: `parse_vacancy()`, `parse_resume()`, `score_pair()`
- Produces: `scoring_graph` — compiled LangGraph, используется в `orchestrator.py` и `sse.py`
- Produces: `score_candidates_batch(vacancy_id, candidate_ids, progress_cb)` — батч с Semaphore(5)

- [ ] **Step 1: Написать `app/pipeline/graph.py`**

```python
import uuid
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from app.schemas.vacancy import ParsedVacancy
from app.schemas.candidate import ParsedResume
from app.schemas.scoring import CandidateScore


class ScoringState(TypedDict):
    vacancy_id: uuid.UUID
    candidate_id: uuid.UUID
    parsed_vacancy: Optional[ParsedVacancy]
    parsed_resume: Optional[ParsedResume]
    score: Optional[CandidateScore]
    error: Optional[str]


async def _parse_vacancy_node(state: ScoringState) -> ScoringState:
    from app.pipeline.parse_vacancy import parse_vacancy
    parsed = await parse_vacancy(state["vacancy_id"])
    return {**state, "parsed_vacancy": parsed}


async def _parse_resume_node(state: ScoringState) -> ScoringState:
    from app.pipeline.parse_resume import parse_resume
    parsed = await parse_resume(state["candidate_id"])
    return {**state, "parsed_resume": parsed}


async def _score_node(state: ScoringState) -> ScoringState:
    from app.pipeline.scorer import score_pair
    result = await score_pair(
        state["vacancy_id"],
        state["candidate_id"],
        state["parsed_vacancy"],
        state["parsed_resume"],
    )
    return {**state, "score": result}


def build_scoring_graph():
    graph = StateGraph(ScoringState)
    graph.add_node("parse_vacancy", _parse_vacancy_node)
    graph.add_node("parse_resume", _parse_resume_node)
    graph.add_node("score", _score_node)
    graph.set_entry_point("parse_vacancy")
    graph.add_edge("parse_vacancy", "parse_resume")
    graph.add_edge("parse_resume", "score")
    graph.add_edge("score", END)
    return graph.compile()


scoring_graph = build_scoring_graph()
```

- [ ] **Step 2: Написать `app/pipeline/orchestrator.py`**

```python
import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Awaitable
from app.pipeline.graph import scoring_graph, ScoringState
from app.schemas.scoring import CandidateScore


async def score_candidates_batch(
    vacancy_id: uuid.UUID,
    candidate_ids: list[uuid.UUID],
    on_progress: Callable[[uuid.UUID, CandidateScore | None, str | None], Awaitable[None]] | None = None,
    max_concurrent: int = 5,
) -> list[tuple[uuid.UUID, CandidateScore]]:
    """
    Запускает scoring для всех кандидатов батчем.
    on_progress(candidate_id, score_or_none, error_or_none) вызывается после каждого кандидата.
    Возвращает список (candidate_id, CandidateScore), отсортированный по score DESC.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(candidate_id: uuid.UUID) -> tuple[uuid.UUID, CandidateScore | None, str | None]:
        async with semaphore:
            state = ScoringState(
                vacancy_id=vacancy_id,
                candidate_id=candidate_id,
                parsed_vacancy=None,
                parsed_resume=None,
                score=None,
                error=None,
            )
            try:
                result_state = await scoring_graph.ainvoke(state)
                score = result_state["score"]
                if on_progress:
                    await on_progress(candidate_id, score, None)
                return candidate_id, score, None
            except Exception as exc:
                error_msg = str(exc)
                if on_progress:
                    await on_progress(candidate_id, None, error_msg)
                return candidate_id, None, error_msg

    tasks = [_run_one(cid) for cid in candidate_ids]
    raw_results = await asyncio.gather(*tasks)

    successful = [
        (cid, score) for cid, score, error in raw_results if score is not None
    ]
    return sorted(successful, key=lambda x: x[1].score, reverse=True)
```

- [ ] **Step 3: Написать тест**

```python
# tests/test_pipeline/test_graph.py
import pytest
import uuid
from unittest.mock import AsyncMock, patch
from app.schemas.vacancy import ParsedVacancy
from app.schemas.candidate import ParsedResume
from app.schemas.scoring import CandidateScore
from app.pipeline.orchestrator import score_candidates_batch

MOCK_VACANCY = ParsedVacancy(
    required_skills=["Python"], nice_to_have=[], min_experience_years=2,
    level="middle", responsibilities=["API development"],
)
MOCK_RESUME = ParsedResume(
    skills=["Python", "FastAPI"], experience_years=3,
    education="CS degree", last_position="Backend @ startup", summary="Python dev",
)
MOCK_SCORE = CandidateScore(
    score=80, grade="strong_match", key_matches=["Python", "3 years"],
    key_gaps=[], summary="Подходящий кандидат.", confidence=0.9,
)

@pytest.mark.asyncio
async def test_score_candidates_batch_returns_sorted():
    vacancy_id = uuid.uuid4()
    candidate_ids = [uuid.uuid4(), uuid.uuid4()]

    scores = [
        CandidateScore(score=60, grade="good_match", key_matches=["Python"],
                       key_gaps=["Опыт меньше"], summary="Неплохо.", confidence=0.8),
        CandidateScore(score=85, grade="strong_match", key_matches=["Python", "FastAPI"],
                       key_gaps=[], summary="Отличный.", confidence=0.95),
    ]

    call_count = 0
    async def mock_ainvoke(state):
        nonlocal call_count
        score = scores[call_count % 2]
        call_count += 1
        return {**state, "parsed_vacancy": MOCK_VACANCY, "parsed_resume": MOCK_RESUME, "score": score}

    with patch("app.pipeline.orchestrator.scoring_graph") as mock_graph:
        mock_graph.ainvoke = mock_ainvoke
        results = await score_candidates_batch(vacancy_id, candidate_ids)

    assert len(results) == 2
    assert results[0][1].score >= results[1][1].score  # sorted DESC
```

- [ ] **Step 4: Запустить тест**

```bash
pytest tests/test_pipeline/test_graph.py -v
# Expected: 1 passed
```

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/graph.py app/pipeline/orchestrator.py tests/test_pipeline/
git commit -m "feat: LangGraph StateGraph scoring workflow + batch orchestrator with Semaphore(5)"
```

---

### Task 7: HH.ru Integration

**Files:**
- Create: `app/integrations/hh/client.py`
- Create: `app/integrations/hh/mapper.py`
- Create: `app/integrations/hh/sync_worker.py`
- Test: `tests/test_integrations/test_hh_client.py`

**Interfaces:**
- Produces: `HHClient` — методы `get_vacancies()`, `get_negotiations(vacancy_id)`, `get_resume(resume_id)`
- Produces: `hh_resume_to_raw_text(hh_resume_json) → str`
- Produces: `sync_hh(db_session) → SyncLog` — записывает вакансии + кандидатов в БД

- [ ] **Step 1: Написать `app/integrations/hh/client.py`**

```python
from typing import Any
import httpx
from app.core.config import settings


class HHClient:
    BASE = "https://api.hh.ru"

    def __init__(self) -> None:
        self._headers = {
            "Authorization": f"Bearer {settings.HH_ACCESS_TOKEN}",
            "HH-User-Agent": "ResumeScoring/1.0 (bogdanatrosenko@gmail.com)",
        }

    async def get_vacancies(self) -> list[dict[str, Any]]:
        """GET /vacancies?employer_id=... — все активные вакансии работодателя."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE}/vacancies",
                params={"employer_id": settings.HH_EMPLOYER_ID, "per_page": 100},
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("items", [])

    async def get_negotiations(self, vacancy_id: str) -> list[dict[str, Any]]:
        """GET /negotiations — отклики по вакансии (список resume_id)."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE}/negotiations",
                params={"vacancy_id": vacancy_id, "per_page": 100, "status": "active"},
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("items", [])

    async def get_resume(self, resume_id: str) -> dict[str, Any]:
        """GET /resumes/{id} — структурированное резюме в JSON."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE}/resumes/{resume_id}",
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_vacancy_detail(self, vacancy_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE}/vacancies/{vacancy_id}",
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
```

- [ ] **Step 2: Написать `app/integrations/hh/mapper.py`**

```python
from typing import Any


def hh_resume_to_raw_text(resume: dict[str, Any]) -> str:
    """Конвертирует структурированный JSON резюме HH.ru в читаемый текст для LLM."""
    parts: list[str] = []

    if title := resume.get("title"):
        parts.append(f"Должность: {title}")

    if exp := resume.get("total_experience"):
        months = exp.get("months", 0)
        years = months // 12
        parts.append(f"Общий опыт: {years} лет ({months} месяцев)")

    if skills := resume.get("skill_set"):
        parts.append(f"Навыки: {', '.join(skills)}")
    elif skills_text := resume.get("skills"):
        parts.append(f"Навыки: {skills_text}")

    experience = resume.get("experience", [])
    if experience:
        parts.append("Опыт работы:")
        for exp_item in experience[:5]:
            company = exp_item.get("company", "")
            position = exp_item.get("position", "")
            start = exp_item.get("start", "")[:7] if exp_item.get("start") else ""
            end = exp_item.get("end", "н.в.")[:7] if exp_item.get("end") else "н.в."
            description = exp_item.get("description", "")[:300]
            parts.append(f"  - {position} @ {company} ({start} — {end}): {description}")

    education = resume.get("education", {})
    primary_edu = education.get("primary", [])
    if primary_edu:
        edu = primary_edu[0]
        parts.append(
            f"Образование: {edu.get('name', '')} — {edu.get('organization', '')} ({edu.get('year', '')})"
        )

    return "\n".join(parts)


def hh_vacancy_to_description(vacancy: dict[str, Any]) -> str:
    """Конвертирует JSON вакансии HH.ru в текстовое описание."""
    parts = []
    if name := vacancy.get("name"):
        parts.append(f"Вакансия: {name}")
    if employer := vacancy.get("employer", {}).get("name"):
        parts.append(f"Компания: {employer}")
    if desc := vacancy.get("description"):
        # HH.ru возвращает HTML — очищаем базово
        import re
        clean = re.sub(r"<[^>]+>", " ", desc)
        clean = re.sub(r"\s+", " ", clean).strip()
        parts.append(f"Описание:\n{clean[:3000]}")
    return "\n".join(parts)
```

- [ ] **Step 3: Написать `app/integrations/hh/sync_worker.py`**

```python
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.vacancy import Vacancy
from app.models.candidate import Candidate
from app.models.sync_log import SyncLog
from app.integrations.hh.client import HHClient
from app.integrations.hh.mapper import hh_resume_to_raw_text, hh_vacancy_to_description
from app.core.config import settings


async def sync_hh(session: AsyncSession) -> SyncLog:
    client = HHClient()
    log = SyncLog(source="hh", status="failed")
    session.add(log)

    try:
        hh_vacancies = await client.get_vacancies()

        for hh_v in hh_vacancies:
            hh_id = str(hh_v["id"])
            existing = (await session.execute(
                select(Vacancy).where(Vacancy.hh_vacancy_id == hh_id)
            )).scalar_one_or_none()

            detail = await client.get_vacancy_detail(hh_id)
            description = hh_vacancy_to_description(detail)

            if existing is None:
                vacancy = Vacancy(
                    hh_vacancy_id=hh_id,
                    title=hh_v.get("name", ""),
                    description=description,
                    employer_name=hh_v.get("employer", {}).get("name"),
                    source="hh",
                )
                session.add(vacancy)
                await session.flush()
            else:
                existing.last_synced_at = datetime.now(timezone.utc)
                vacancy = existing

            log.vacancy_id = vacancy.id

            negotiations = await client.get_negotiations(hh_id)
            log.candidates_fetched += len(negotiations)

            for neg in negotiations:
                resume_data = neg.get("resume")
                if not resume_data:
                    continue
                resume_id = str(resume_data["id"])

                existing_c = (await session.execute(
                    select(Candidate).where(Candidate.hh_resume_id == resume_id)
                )).scalar_one_or_none()

                if existing_c is None:
                    full_resume = await client.get_resume(resume_id)
                    candidate = Candidate(
                        hh_resume_id=resume_id,
                        name=full_resume.get("last_name", "") + " " + full_resume.get("first_name", ""),
                        raw_text=hh_resume_to_raw_text(full_resume),
                        source="hh",
                    )
                    session.add(candidate)
                    log.candidates_new += 1

        log.status = "success"
        log.finished_at = datetime.now(timezone.utc)
        await session.commit()

    except Exception as exc:
        log.status = "failed"
        log.error_message = str(exc)[:500]
        log.finished_at = datetime.now(timezone.utc)
        await session.commit()
        raise

    return log
```

- [ ] **Step 4: Написать тест (мокируем httpx)**

```python
# tests/test_integrations/test_hh_client.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.integrations.hh.client import HHClient
from app.integrations.hh.mapper import hh_resume_to_raw_text, hh_vacancy_to_description

def test_hh_resume_to_raw_text_extracts_skills():
    resume = {
        "title": "Python Developer",
        "total_experience": {"months": 36},
        "skill_set": ["Python", "FastAPI", "PostgreSQL"],
        "experience": [
            {"company": "Tinkoff", "position": "Backend", "start": "2022-01-01", "end": None}
        ],
        "education": {"primary": [{"name": "МГТУ", "organization": "ФН", "year": 2020}]},
    }
    text = hh_resume_to_raw_text(resume)
    assert "Python" in text
    assert "3 лет" in text
    assert "Tinkoff" in text

def test_hh_vacancy_to_description():
    vacancy = {
        "name": "Python Backend",
        "employer": {"name": "Acme Corp"},
        "description": "<p>Нужен Python разработчик.</p>",
    }
    text = hh_vacancy_to_description(vacancy)
    assert "Python Backend" in text
    assert "Acme Corp" in text
    assert "<p>" not in text

@pytest.mark.asyncio
async def test_hh_client_get_vacancies_returns_items():
    client = HHClient()
    mock_response = MagicMock()
    mock_response.json.return_value = {"items": [{"id": "123", "name": "Dev"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_http:
        mock_http.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        result = await client.get_vacancies()

    assert len(result) == 1
    assert result[0]["id"] == "123"
```

- [ ] **Step 5: Запустить тест**

```bash
pytest tests/test_integrations/test_hh_client.py -v
# Expected: 3 passed
```

- [ ] **Step 6: Commit**

```bash
git add app/integrations/hh/ tests/test_integrations/test_hh_client.py
git commit -m "feat: HH.ru employer integration — client, mapper, sync worker"
```

---

### Task 8: SuperJob Integration (2nd Source)

**Files:**
- Create: `app/integrations/superjob/client.py`
- Create: `app/integrations/superjob/mapper.py`
- Create: `app/integrations/superjob/sync_worker.py`
- Test: `tests/test_integrations/test_superjob_client.py`

**Interfaces:**
- Produces: `SuperJobClient` — методы `get_vacancies()`, `get_responses(vacancy_id)`, `get_resume(resume_id)`
- Produces: `sj_resume_to_raw_text(sj_resume) → str`
- Produces: `sync_superjob(db_session) → SyncLog`

- [ ] **Step 1: Написать `app/integrations/superjob/client.py`**

```python
from typing import Any
import httpx
from app.core.config import settings


class SuperJobClient:
    BASE = "https://api.superjob.ru/2.0"

    def __init__(self) -> None:
        self._headers = {
            "X-Api-App-Id": settings.SUPERJOB_API_KEY,
            "Authorization": f"Bearer {settings.SUPERJOB_API_KEY}",
        }

    async def get_vacancies(self) -> list[dict[str, Any]]:
        """GET /2.0/vacancies/?firm_id=... — вакансии компании."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE}/vacancies/",
                params={"firm_id": settings.SUPERJOB_EMPLOYER_ID, "count": 100, "archive": 0},
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("objects", [])

    async def get_responses(self, vacancy_id: str) -> list[dict[str, Any]]:
        """GET /2.0/resumes/?vacancy_id=... — отклики на вакансию."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE}/resumes/",
                params={"vacancy_id": vacancy_id, "count": 100},
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("objects", [])

    async def get_resume(self, resume_id: str) -> dict[str, Any]:
        """GET /2.0/resumes/{id}/ — полное резюме кандидата."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.BASE}/resumes/{resume_id}/",
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
```

- [ ] **Step 2: Написать `app/integrations/superjob/mapper.py`**

```python
from typing import Any


def sj_resume_to_raw_text(resume: dict[str, Any]) -> str:
    parts: list[str] = []

    if position := resume.get("profession"):
        parts.append(f"Должность: {position}")

    exp_years = resume.get("experience", {}).get("value", 0) // 12
    if exp_years:
        parts.append(f"Опыт: {exp_years} лет")

    if skills := resume.get("skills"):
        parts.append(f"Навыки: {skills}")

    for job in resume.get("experience_list", [])[:5]:
        company = job.get("firm_name", "")
        position = job.get("position", "")
        date_from = job.get("date_from", "")
        date_to = job.get("date_to", "н.в.")
        resp = job.get("duties", "")[:300]
        parts.append(f"  - {position} @ {company} ({date_from} — {date_to}): {resp}")

    for edu in resume.get("education", {}).get("items", [])[:1]:
        parts.append(f"Образование: {edu.get('name', '')} ({edu.get('end_year', '')})")

    return "\n".join(parts)


def sj_vacancy_to_description(vacancy: dict[str, Any]) -> str:
    parts = []
    if name := vacancy.get("profession"):
        parts.append(f"Вакансия: {name}")
    if firm := vacancy.get("firm_name"):
        parts.append(f"Компания: {firm}")
    if desc := vacancy.get("candidat"):
        parts.append(f"Требования:\n{desc[:3000]}")
    if work := vacancy.get("work"):
        parts.append(f"Обязанности:\n{work[:2000]}")
    return "\n".join(parts)
```

- [ ] **Step 3: Написать `app/integrations/superjob/sync_worker.py`**

```python
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.vacancy import Vacancy
from app.models.candidate import Candidate
from app.models.sync_log import SyncLog
from app.integrations.superjob.client import SuperJobClient
from app.integrations.superjob.mapper import sj_resume_to_raw_text, sj_vacancy_to_description


async def sync_superjob(session: AsyncSession) -> SyncLog:
    client = SuperJobClient()
    log = SyncLog(source="superjob", status="failed")
    session.add(log)

    try:
        sj_vacancies = await client.get_vacancies()

        for sj_v in sj_vacancies:
            sj_id = str(sj_v["id"])
            existing = (await session.execute(
                select(Vacancy).where(Vacancy.superjob_vacancy_id == sj_id)
            )).scalar_one_or_none()

            description = sj_vacancy_to_description(sj_v)

            if existing is None:
                vacancy = Vacancy(
                    superjob_vacancy_id=sj_id,
                    title=sj_v.get("profession", ""),
                    description=description,
                    employer_name=sj_v.get("firm_name"),
                    salary_from=sj_v.get("payment_from"),
                    salary_to=sj_v.get("payment_to"),
                    salary_currency="RUB",
                    source="superjob",
                )
                session.add(vacancy)
                await session.flush()
            else:
                existing.last_synced_at = datetime.now(timezone.utc)
                vacancy = existing

            log.vacancy_id = vacancy.id

            responses = await client.get_responses(sj_id)
            log.candidates_fetched += len(responses)

            for res in responses:
                sj_resume_id = str(res["id"])
                existing_c = (await session.execute(
                    select(Candidate).where(Candidate.superjob_resume_id == sj_resume_id)
                )).scalar_one_or_none()

                if existing_c is None:
                    candidate = Candidate(
                        superjob_resume_id=sj_resume_id,
                        name=f"{res.get('lastName', '')} {res.get('firstName', '')}".strip(),
                        raw_text=sj_resume_to_raw_text(res),
                        source="superjob",
                    )
                    session.add(candidate)
                    log.candidates_new += 1

        log.status = "success"
        log.finished_at = datetime.now(timezone.utc)
        await session.commit()

    except Exception as exc:
        log.status = "failed"
        log.error_message = str(exc)[:500]
        log.finished_at = datetime.now(timezone.utc)
        await session.commit()
        raise

    return log
```

- [ ] **Step 4: Написать тест**

```python
# tests/test_integrations/test_superjob_client.py
from app.integrations.superjob.mapper import sj_resume_to_raw_text, sj_vacancy_to_description

def test_sj_resume_mapper_extracts_profession():
    resume = {
        "profession": "Python Developer",
        "experience": {"value": 48},
        "skills": "Python, FastAPI",
        "experience_list": [
            {"firm_name": "Yandex", "position": "Backend", "date_from": "2021", "duties": "API dev"}
        ],
    }
    text = sj_resume_to_raw_text(resume)
    assert "Python Developer" in text
    assert "4 лет" in text
    assert "Yandex" in text

def test_sj_vacancy_mapper():
    vacancy = {
        "profession": "Go Developer",
        "firm_name": "Acme",
        "candidat": "Опыт Go от 3 лет",
        "work": "Разработка микросервисов",
    }
    text = sj_vacancy_to_description(vacancy)
    assert "Go Developer" in text
    assert "Acme" in text
```

- [ ] **Step 5: Запустить тест**

```bash
pytest tests/test_integrations/test_superjob_client.py -v
# Expected: 2 passed
```

- [ ] **Step 6: Commit**

```bash
git add app/integrations/superjob/ tests/test_integrations/test_superjob_client.py
git commit -m "feat: SuperJob integration — client, mapper, sync worker"
```

---

### Task 9: APScheduler + Trigger Batch Scoring

**Files:**
- Create: `app/scheduler.py`
- Modify: `app/main.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `sync_hh()`, `sync_superjob()`, `score_candidates_batch()`
- Produces: APScheduler instance запускается при старте FastAPI через lifespan

- [ ] **Step 1: Написать `app/scheduler.py`**

```python
import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.core.config import settings
from app.core.database import get_session

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()


async def _run_full_sync() -> None:
    """Синхронизирует HH.ru и SuperJob, затем запускает scoring для новых кандидатов."""
    from app.integrations.hh.sync_worker import sync_hh
    from app.integrations.superjob.sync_worker import sync_superjob
    from app.pipeline.orchestrator import score_candidates_batch
    from app.models.candidate import Candidate
    from app.models.scoring_result import ScoringResult
    from app.models.vacancy import Vacancy
    from sqlalchemy import select

    logger.info("Starting scheduled sync")

    async with get_session() as session:
        try:
            await sync_hh(session)
        except Exception as exc:
            logger.error(f"HH sync failed: {exc}")

        try:
            await sync_superjob(session)
        except Exception as exc:
            logger.error(f"SuperJob sync failed: {exc}")

        # Score all unscored pairs
        vacancies = (await session.execute(select(Vacancy))).scalars().all()
        for vacancy in vacancies:
            all_candidates = (await session.execute(select(Candidate))).scalars().all()
            scored_ids = set(
                row.candidate_id for row in (
                    await session.execute(
                        select(ScoringResult.candidate_id).where(
                            ScoringResult.vacancy_id == vacancy.id
                        )
                    )
                ).scalars().all()
            )
            unscored = [c.id for c in all_candidates if c.id not in scored_ids]
            if unscored:
                logger.info(f"Scoring {len(unscored)} candidates for vacancy {vacancy.id}")
                await score_candidates_batch(vacancy.id, unscored)

    logger.info("Scheduled sync complete")


def setup_scheduler() -> None:
    scheduler.add_job(
        _run_full_sync,
        trigger=IntervalTrigger(hours=settings.SYNC_INTERVAL_HOURS),
        id="full_sync",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
```

- [ ] **Step 2: Написать `app/main.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.scheduler import scheduler, setup_scheduler
from app.web.routes import vacancies, candidates, upload, sse, api


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_scheduler()
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="AI Resume Assessment", lifespan=lifespan)

app.include_router(vacancies.router)
app.include_router(candidates.router)
app.include_router(upload.router)
app.include_router(sse.router)
app.include_router(api.router, prefix="/api")
```

- [ ] **Step 3: Написать тест**

```python
# tests/test_scheduler.py
from app.scheduler import scheduler, setup_scheduler

def test_setup_scheduler_registers_job():
    setup_scheduler()
    job = scheduler.get_job("full_sync")
    assert job is not None
    assert job.id == "full_sync"
```

- [ ] **Step 4: Запустить тест**

```bash
pytest tests/test_scheduler.py -v
# Expected: 1 passed
```

- [ ] **Step 5: Commit**

```bash
git add app/scheduler.py app/main.py tests/test_scheduler.py
git commit -m "feat: APScheduler — full sync job every 6h, FastAPI lifespan integration"
```

---

### Task 10: Web UI — Vacancy List & Candidate Rankings

**Files:**
- Create: `app/web/routes/vacancies.py`
- Create: `app/web/routes/candidates.py`
- Create: `app/templates/base.html`
- Create: `app/templates/vacancies/list.html`
- Create: `app/templates/vacancies/detail.html`
- Create: `app/templates/candidates/detail.html`
- Test: `tests/test_web/test_vacancies.py`

**Interfaces:**
- Consumes: `Vacancy`, `Candidate`, `ScoringResult` из DB
- Produces: GET `/vacancies` → список вакансий
- Produces: GET `/vacancies/{id}` → вакансия + кандидаты, отсортированные по score DESC
- Produces: GET `/candidates/{id}` → кандидат + детали оценки

- [ ] **Step 1: Написать `app/web/routes/vacancies.py`**

```python
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, desc
from app.core.database import get_session
from app.models.vacancy import Vacancy
from app.models.candidate import Candidate
from app.models.scoring_result import ScoringResult

router = APIRouter(tags=["vacancies"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def vacancy_list(request: Request):
    async with get_session() as session:
        vacancies = (await session.execute(select(Vacancy).order_by(desc(Vacancy.created_at)))).scalars().all()
    return templates.TemplateResponse("vacancies/list.html", {"request": request, "vacancies": vacancies})


@router.get("/vacancies/{vacancy_id}", response_class=HTMLResponse)
async def vacancy_detail(request: Request, vacancy_id: uuid.UUID):
    async with get_session() as session:
        vacancy = (await session.execute(select(Vacancy).where(Vacancy.id == vacancy_id))).scalar_one()
        results = (await session.execute(
            select(ScoringResult, Candidate)
            .join(Candidate, ScoringResult.candidate_id == Candidate.id)
            .where(ScoringResult.vacancy_id == vacancy_id)
            .order_by(desc(ScoringResult.score))
        )).all()

    ranked = [
        {
            "rank": i + 1,
            "candidate": candidate,
            "result": result,
        }
        for i, (result, candidate) in enumerate(results)
    ]
    return templates.TemplateResponse(
        "vacancies/detail.html",
        {"request": request, "vacancy": vacancy, "ranked": ranked},
    )
```

- [ ] **Step 2: Написать `app/web/routes/candidates.py`**

```python
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from app.core.database import get_session
from app.models.candidate import Candidate
from app.models.scoring_result import ScoringResult
from app.models.vacancy import Vacancy

router = APIRouter(tags=["candidates"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/candidates/{candidate_id}", response_class=HTMLResponse)
async def candidate_detail(request: Request, candidate_id: uuid.UUID):
    async with get_session() as session:
        candidate = (await session.execute(
            select(Candidate).where(Candidate.id == candidate_id)
        )).scalar_one()

        scoring_rows = (await session.execute(
            select(ScoringResult, Vacancy)
            .join(Vacancy, ScoringResult.vacancy_id == Vacancy.id)
            .where(ScoringResult.candidate_id == candidate_id)
        )).all()

    assessments = [
        {"result": result, "vacancy": vacancy}
        for result, vacancy in scoring_rows
    ]
    return templates.TemplateResponse(
        "candidates/detail.html",
        {"request": request, "candidate": candidate, "assessments": assessments},
    )
```

- [ ] **Step 3: Написать `app/templates/base.html`**

```html
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{% block title %}AI Resume Assessment{% endblock %}</title>
  <script src="https://unpkg.com/htmx.org@2.0.3"></script>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 1rem; }
    nav a { margin-right: 1.5rem; text-decoration: none; color: #2563eb; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 0.5rem 0.75rem; border: 1px solid #e5e7eb; text-align: left; }
    th { background: #f9fafb; }
    .badge { padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .strong_match { background: #d1fae5; color: #065f46; }
    .good_match   { background: #dbeafe; color: #1e40af; }
    .weak_match   { background: #fef3c7; color: #92400e; }
    .no_match     { background: #fee2e2; color: #991b1b; }
  </style>
</head>
<body>
  <nav>
    <a href="/">Вакансии</a>
    <a href="/upload">Ручная загрузка</a>
  </nav>
  <hr/>
  {% block content %}{% endblock %}
</body>
</html>
```

- [ ] **Step 4: Написать `app/templates/vacancies/list.html`**

```html
{% extends "base.html" %}
{% block title %}Вакансии{% endblock %}
{% block content %}
<h1>Вакансии компании</h1>
<table>
  <thead>
    <tr><th>Название</th><th>Источник</th><th>Статус</th><th>Создана</th></tr>
  </thead>
  <tbody>
    {% for v in vacancies %}
    <tr>
      <td><a href="/vacancies/{{ v.id }}">{{ v.title }}</a></td>
      <td>{{ v.source }}</td>
      <td>{{ v.status }}</td>
      <td>{{ v.created_at.strftime('%d.%m.%Y') }}</td>
    </tr>
    {% else %}
    <tr><td colspan="4">Вакансии не найдены. Запустите синхронизацию или загрузите вручную.</td></tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

- [ ] **Step 5: Написать `app/templates/vacancies/detail.html`**

```html
{% extends "base.html" %}
{% block title %}{{ vacancy.title }}{% endblock %}
{% block content %}
<h1>{{ vacancy.title }}</h1>
<p><strong>Источник:</strong> {{ vacancy.source }} | <strong>Статус:</strong> {{ vacancy.status }}</p>

<h2>Кандидаты ({{ ranked | length }}), по убыванию оценки</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>Имя</th><th>Оценка</th><th>Грейд</th>
      <th>Преимущества</th><th>Пробелы</th><th>Выжимка</th>
    </tr>
  </thead>
  <tbody>
    {% for row in ranked %}
    <tr>
      <td>{{ row.rank }}</td>
      <td><a href="/candidates/{{ row.candidate.id }}">{{ row.candidate.name or 'Аноним' }}</a></td>
      <td><strong>{{ row.result.score }}</strong></td>
      <td><span class="badge {{ row.result.grade }}">{{ row.result.grade.replace('_', ' ') }}</span></td>
      <td>{{ row.result.key_matches | join(', ') }}</td>
      <td>{{ row.result.key_gaps | join(', ') or '—' }}</td>
      <td>{{ row.result.summary }}</td>
    </tr>
    {% else %}
    <tr><td colspan="7">Оценки ещё не рассчитаны.</td></tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

- [ ] **Step 6: Написать `app/templates/candidates/detail.html`**

```html
{% extends "base.html" %}
{% block title %}{{ candidate.name or 'Кандидат' }}{% endblock %}
{% block content %}
<h1>{{ candidate.name or 'Кандидат без имени' }}</h1>
<p><strong>Источник:</strong> {{ candidate.source }}</p>

{% if candidate.parsed %}
<h2>Распарсенный профиль</h2>
<ul>
  <li><strong>Навыки:</strong> {{ candidate.parsed.skills | join(', ') }}</li>
  <li><strong>Опыт:</strong> {{ candidate.parsed.experience_years }} лет</li>
  <li><strong>Последняя должность:</strong> {{ candidate.parsed.last_position }}</li>
  <li><strong>Образование:</strong> {{ candidate.parsed.education }}</li>
</ul>
{% endif %}

<h2>Оценки по вакансиям</h2>
{% for a in assessments %}
<div style="border:1px solid #e5e7eb; border-radius:6px; padding:1rem; margin-bottom:1rem;">
  <h3><a href="/vacancies/{{ a.vacancy.id }}">{{ a.vacancy.title }}</a></h3>
  <p>
    <strong>Оценка:</strong> {{ a.result.score }}/100 &nbsp;
    <span class="badge {{ a.result.grade }}">{{ a.result.grade.replace('_', ' ') }}</span>
    &nbsp; <strong>Уверенность:</strong> {{ (a.result.confidence * 100) | round | int }}%
  </p>
  <p><strong>Преимущества:</strong>
    <ul>{% for m in a.result.key_matches %}<li>{{ m }}</li>{% endfor %}</ul>
  </p>
  {% if a.result.key_gaps %}
  <p><strong>Пробелы:</strong>
    <ul>{% for g in a.result.key_gaps %}<li>{{ g }}</li>{% endfor %}</ul>
  </p>
  {% endif %}
  <p><em>{{ a.result.summary }}</em></p>
</div>
{% else %}
<p>Нет оценок.</p>
{% endfor %}
{% endblock %}
```

- [ ] **Step 7: Написать тест**

```python
# tests/test_web/test_vacancies.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
import uuid
from datetime import datetime, timezone
from app.main import app

MOCK_VACANCY = MagicMock()
MOCK_VACANCY.id = uuid.uuid4()
MOCK_VACANCY.title = "Python Dev"
MOCK_VACANCY.source = "hh"
MOCK_VACANCY.status = "active"
MOCK_VACANCY.created_at = datetime.now(timezone.utc)

@pytest.mark.asyncio
async def test_vacancy_list_returns_200():
    with patch("app.web.routes.vacancies.get_session") as mock_gs:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[MOCK_VACANCY])))))
        mock_gs.return_value = mock_session

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/")

    assert resp.status_code == 200
    assert "Python Dev" in resp.text
```

- [ ] **Step 8: Запустить тест**

```bash
pytest tests/test_web/test_vacancies.py -v
# Expected: 1 passed
```

- [ ] **Step 9: Commit**

```bash
git add app/web/routes/vacancies.py app/web/routes/candidates.py app/templates/ tests/test_web/
git commit -m "feat: web UI — vacancy list, candidate ranking table, candidate detail with score breakdown"
```

---

### Task 11: Manual Upload Mode + SSE Progress

**Files:**
- Create: `app/web/routes/upload.py`
- Create: `app/web/routes/sse.py`
- Create: `app/web/routes/api.py`
- Create: `app/templates/upload.html`
- Test: `tests/test_web/test_upload.py`

**Interfaces:**
- Consumes: `score_candidates_batch()`, `parse_resume()` (для PDF через pdfminer)
- Produces: POST `/upload` → сохраняет Vacancy + Candidates, запускает batch scoring
- Produces: GET `/score/progress/{job_id}` → SSE stream с JSON событиями `{done, total, candidate_id, score}`
- Produces: POST `/api/score` → триггер batch scoring (используется HTMX)

- [ ] **Step 1: Написать `app/web/routes/upload.py`**

```python
import uuid
import asyncio
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pdfminer.high_level import extract_text
from io import BytesIO
from app.core.database import get_session
from app.models.vacancy import Vacancy
from app.models.candidate import Candidate
from app.pipeline.orchestrator import score_candidates_batch

router = APIRouter(tags=["upload"])
templates = Jinja2Templates(directory="app/templates")

# In-memory job progress store (достаточно для MVP)
_job_progress: dict[str, dict] = {}


def _extract_pdf_text(file_bytes: bytes) -> str:
    return extract_text(BytesIO(file_bytes))


@router.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@router.post("/upload", response_class=HTMLResponse)
async def upload_submit(
    request: Request,
    vacancy_text: str = Form(...),
    resumes: list[UploadFile] = File(...),
):
    job_id = str(uuid.uuid4())
    _job_progress[job_id] = {"done": 0, "total": len(resumes), "results": []}

    async with get_session() as session:
        vacancy = Vacancy(title="Ручная загрузка", description=vacancy_text, source="manual")
        session.add(vacancy)
        await session.flush()

        candidate_ids = []
        for resume_file in resumes:
            file_bytes = await resume_file.read()
            if resume_file.filename.endswith(".pdf"):
                raw_text = _extract_pdf_text(file_bytes)
            else:
                raw_text = file_bytes.decode("utf-8", errors="replace")

            candidate = Candidate(
                name=resume_file.filename.replace(".pdf", "").replace(".txt", ""),
                raw_text=raw_text,
                source="manual",
            )
            session.add(candidate)
            await session.flush()
            candidate_ids.append(candidate.id)

        vacancy_id = vacancy.id

    async def _progress_cb(candidate_id, score, error):
        _job_progress[job_id]["done"] += 1
        if score:
            _job_progress[job_id]["results"].append({
                "candidate_id": str(candidate_id),
                "score": score.score,
                "grade": score.grade,
            })

    asyncio.create_task(score_candidates_batch(vacancy_id, candidate_ids, on_progress=_progress_cb))

    return RedirectResponse(f"/vacancies/{vacancy_id}?job_id={job_id}", status_code=303)
```

- [ ] **Step 2: Написать `app/web/routes/sse.py`**

```python
import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.web.routes.upload import _job_progress

router = APIRouter(tags=["sse"])


@router.get("/score/progress/{job_id}")
async def score_progress(job_id: str):
    async def event_stream():
        while True:
            progress = _job_progress.get(job_id, {"done": 0, "total": 0})
            data = json.dumps(progress)
            yield f"data: {data}\n\n"
            if progress["done"] >= progress.get("total", 0) > 0:
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 3: Написать `app/web/routes/api.py`**

```python
import uuid
from fastapi import APIRouter
from sqlalchemy import select
from app.core.database import get_session
from app.models.candidate import Candidate
from app.models.scoring_result import ScoringResult
from app.pipeline.orchestrator import score_candidates_batch
import asyncio

router = APIRouter(tags=["api"])


@router.post("/score/{vacancy_id}")
async def trigger_scoring(vacancy_id: uuid.UUID):
    """HTMX-дружественный endpoint: запускает scoring для всех нескоренных кандидатов вакансии."""
    async with get_session() as session:
        all_candidates = (await session.execute(select(Candidate))).scalars().all()
        scored_ids = set(
            (await session.execute(
                select(ScoringResult.candidate_id).where(ScoringResult.vacancy_id == vacancy_id)
            )).scalars().all()
        )
        unscored = [c.id for c in all_candidates if c.id not in scored_ids]

    if unscored:
        asyncio.create_task(score_candidates_batch(vacancy_id, unscored))
        return {"status": "started", "candidates": len(unscored)}
    return {"status": "nothing_to_score"}
```

- [ ] **Step 4: Написать `app/templates/upload.html`**

```html
{% extends "base.html" %}
{% block title %}Ручная загрузка{% endblock %}
{% block content %}
<h1>Ручная оценка кандидатов</h1>
<form method="post" action="/upload" enctype="multipart/form-data">
  <div style="margin-bottom:1rem;">
    <label><strong>Описание вакансии</strong></label><br/>
    <textarea name="vacancy_text" rows="10" style="width:100%;"
      placeholder="Вставьте описание вакансии..."></textarea>
  </div>
  <div style="margin-bottom:1rem;">
    <label><strong>Резюме кандидатов</strong> (PDF или TXT, мультизагрузка)</label><br/>
    <input type="file" name="resumes" multiple accept=".pdf,.txt"/>
  </div>
  <button type="submit" style="padding:0.5rem 1.5rem; background:#2563eb; color:white; border:none; border-radius:4px; cursor:pointer;">
    Оценить
  </button>
</form>
{% endblock %}
```

- [ ] **Step 5: Написать тест**

```python
# tests/test_web/test_upload.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_upload_form_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/upload")
    assert resp.status_code == 200
    assert "Описание вакансии" in resp.text
```

- [ ] **Step 6: Запустить тест**

```bash
pytest tests/test_web/test_upload.py -v
# Expected: 1 passed
```

- [ ] **Step 7: Commit**

```bash
git add app/web/routes/upload.py app/web/routes/sse.py app/web/routes/api.py \
  app/templates/upload.html tests/test_web/test_upload.py
git commit -m "feat: manual upload mode (PDF/TXT), SSE progress stream, API scoring trigger"
```

---

### Task 12: Subagent Decomposition Strategy

Этот раздел описывает, как разбить реализацию плана на параллельных субагентов.

**Предложение: 5 субагентов, работают последовательно по фазам**

---

#### Субагент 1 — Infrastructure & DB
**Задачи плана:** Task 1, Task 2
**Что получает:** `pyproject.toml` с deps, docker-compose, `app/core/config.py`
**Что производит:** Запущенная БД, все ORM-модели, Alembic migration, `get_session()`
**Верификация:** `pytest tests/test_config.py tests/test_models.py` — все зелёные; `docker compose up db -d && alembic upgrade head` без ошибок

---

#### Субагент 2 — LLM Layer
**Задачи плана:** Task 3, Task 4, Task 5
**Зависит от:** Субагент 1 (нужны модели + `get_session`)
**Что получает:** `app/models/`, `app/core/`, `app/schemas/`
**Что производит:** `app/llm/` (все клиенты + factory), `app/pipeline/prompts.py`, `parse_vacancy.py`, `parse_resume.py`, `scorer.py`
**Верификация:** `pytest tests/test_llm/ -v` — все зелёные

---

#### Субагент 3 — Integrations & Scheduler
**Задачи плана:** Task 6, Task 7, Task 8, Task 9
**Зависит от:** Субагент 1 (модели), Субагент 2 (orchestrator)
**Что получает:** `app/integrations/hh/`, `app/integrations/superjob/`, `app/pipeline/graph.py`, `app/pipeline/orchestrator.py`
**Что производит:** оба sync workers, `app/scheduler.py`, `app/main.py` (lifespan)
**Верификация:** `pytest tests/test_integrations/ tests/test_pipeline/ tests/test_scheduler.py -v`

---

#### Субагент 4 — Web UI
**Задачи плана:** Task 10, Task 11
**Зависит от:** Субагент 1 (модели), Субагент 3 (orchestrator готов)
**Что получает:** `app/web/routes/`, `app/templates/base.html`
**Что производит:** все routes (`vacancies`, `candidates`, `upload`, `sse`, `api`), все templates
**Верификация:** `pytest tests/test_web/ -v`; `uvicorn app.main:app --reload` + открыть `http://localhost:8000`

---

#### Субагент 5 — Integration Test & Smoke
**Задачи плана:** финальная верификация (не отдельный task, но нужен после всех)
**Зависит от:** Субагенты 1–4 завершены
**Что делает:**
1. `docker compose up --build -d`
2. `alembic upgrade head`
3. Проверяет `GET /` → список вакансий (200)
4. `POST /upload` с тестовым PDF → редирект на страницу вакансии
5. Убеждается что `scoring_results` в БД заполнены
6. Проверяет `GET /score/progress/{job_id}` отдаёт SSE
7. `pytest tests/ -v` — всё зелёное

---

## Self-Review

### Покрытие спеки

| Требование | Task |
|---|---|
| Scheduler каждые 6ч | Task 9 |
| HH.ru OAuth2 + структурированный JSON | Task 7 |
| SuperJob (2й источник) | Task 8 |
| Parse Vacancy (LLM Call 1) | Task 5 |
| Parse Resume (LLM Call 2) | Task 5 |
| Score (LLM Call 3) | Task 5 |
| Идемпотентность (кэш = БД) | Task 5 (проверки parsed + UNIQUE) |
| Ранжированная таблица кандидатов | Task 10 |
| Карточка кандидата с оценкой | Task 10 |
| Ручной режим (PDF/TXT upload) | Task 11 |
| SSE прогресс при батч-обработке | Task 11 |
| Абстрактный LLMClient (DeepSeek/OpenRouter/Anthropic) | Task 3 |
| temperature=0, function calling | Task 3 (все клиенты) |
| Docker Compose | Task 1 |
| Alembic миграции | Task 2 |
| 4 таблицы БД | Task 2 |
| LangGraph workflow | Task 6 |
| Semaphore(5) | Task 6 (orchestrator) |
| key_gaps=[] для идеального кандидата | Task 5 (prompt), Task 4 (schema) |
| model_used в scoring_results | Task 5 (scorer.py) |

### Placeholder scan — не найдено. Весь код написан полностью.

### Type consistency
- `parse_vacancy(vacancy_id: uuid.UUID) → ParsedVacancy` — используется в `_parse_vacancy_node` в graph.py ✓
- `score_pair(vacancy_id, candidate_id, parsed_vacancy, parsed_resume)` — сигнатура в scorer.py совпадает с вызовом в `_score_node` ✓
- `ScoringState["score"]` — тип `Optional[CandidateScore]` — возвращается в web routes ✓
- `score_candidates_batch(vacancy_id, candidate_ids, on_progress, max_concurrent)` — используется в upload.py и api.py ✓
