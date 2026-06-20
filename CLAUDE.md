# CLAUDE.md — Правила проекта для модели

## Стек (не обсуждается)

| Компонент | Выбор |
|---|---|
| Язык | Python 3.12 |
| Менеджер зависимостей | **Poetry** (не pip, не uv) |
| Конфиг/секреты | **pydantic-settings** + `.env` (не `os.environ` напрямую) |
| Backend | FastAPI (async) |
| UI | FastAPI + Jinja2 + HTMX (не Streamlit, не React) |
| БД | PostgreSQL + SQLAlchemy (async) + Alembic |
| Инфраструктура | Docker Compose (app + postgres + nginx) |
| LLM | Anthropic Claude API через официальный SDK |
| LLM output | tool_use API → Pydantic validated schema (не парсинг текста) |

## LLM-параметры

- Провайдер: `LLM_PROVIDER=deepseek|openrouter|anthropic` (дефолт: `deepseek`)
- Модель: `LLM_MODEL` env var (дефолт зависит от провайдера: `deepseek-chat` / `claude-haiku-4-5-20251001`)
- `temperature=0` для детерминированного скоринга
- Structured output только через function calling / tool_use → Pydantic model
- Никогда не парсить JSON из сырого текстового ответа
- LLMClient — абстрактный Protocol: DeepSeekClient | OpenRouterClient | AnthropicClient

## Архитектурные ограничения

### Scoring Module
- Двухшаговый пайплайн: **Вызов 1 (Extract)** отдельно от **Вызова 2 (Score)**
- Resume Parser кэширует ParsedResume по `hash(file_bytes)`
- Batch Orchestrator использует `asyncio.gather` + `Semaphore(5)` + exponential backoff
- SSE-прогресс обязателен при обработке > 1 резюме

### Общие правила
- Secrets только через pydantic-settings из `.env` — никогда хардкодом
- Резюме и job description не логируются в stdout (персональные данные)
- Каждый модуль имеет одну ответственность и понятный интерфейс (вход/выход)

## Подход к разработке

1. **Сначала диаграмма** (`docs/*.drawio`) — потом файловая структура — потом код
2. **Последовательная разработка модулей**: Scoring Module, затем Market Module. Сразу не пытайся сделать оба
3. **Несогласия фиксируются** в `docs/disagreements.md` (что предложил Claude, почему отвергли, что сделали)
4. Перед крупной реализацией — опираемся на `superpowers:brainstorming` или `/plan` mode
5. После завершения модуля — `/code-review`

## Стиль кода

- Type hints обязательны везде
- Pydantic модели для всех входных/выходных данных
- Async везде где есть I/O (DB, HTTP, filesystem)
- Тесты: pytest + pytest-asyncio; фикстуры в `tests/fixtures/`
- Eval harness: `evals/` (метрики качества LLM-скоринга)

## Чего НЕ делать

- Не предлагать pip/uv вместо Poetry
- Не парсить LLM-ответ из сырого текста — только tool_use → Pydantic
- Не смешивать разработку Scoring Module и Market Module в одной итерации
- Не добавлять фичи "на будущее" — только то, что нужно сейчас (YAGNI)
- Не логировать чувствительные данные (резюме, ключи API)
- Не упрощать до Streamlit — UI строго на Jinja2 + HTMX

## Структура репозитория

```
docs/
  scoring-module.drawio    # Диаграмма Scoring Module
  market-module.drawio     # Диаграмма Market Module (после Scoring)
  disagreements.md         # Лог несогласий с Claude
  ai-collaboration.md      # 5-10 ключевых эпизодов AI-коллаборации
  superpowers/specs/       # Спецификации модулей (brainstorming output)
app/                       # FastAPI приложение
evals/                     # Evaluation harness
tests/                     # Unit + integration тесты
```
