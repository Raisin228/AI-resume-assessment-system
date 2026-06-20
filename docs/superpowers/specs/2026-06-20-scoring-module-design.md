# Scoring Module — Design Spec

**Дата:** 2026-06-20  
**Статус:** In Review (обновлена после обсуждения)  
**Следующий шаг:** writing-plans → реализация

---

## Контекст

Scoring Module — ядро системы AI Resume Assessment. Решает главную боль HR:
система **автоматически** получает описание вакансии и откликнувшихся кандидатов
из HH.ru, периодически запускает LLM-оценку новых откликов и возвращает рекрутёру
ранжированный список с оценкой (0–100), категорией соответствия, топ-3 преимуществами,
топ-3 пробелами (пустые если кандидат идеальный) и кратким резюме.

**Ключевое отличие от ручного подхода:** рекрутёр не загружает файлы вручную —
система сама подгружает резюме из HH.ru откликов по расписанию.

---

## Два режима работы

### Режим 1: Автоматический (основной) — HH.ru интеграция

```
Scheduler (периодически)
  → HH.ru Employer API (OAuth2)
  → GET /vacancies?employer_id=...         → список активных вакансий
  → GET /negotiations?vacancy_id=...       → список откликов (resume_id)
  → GET /resumes/{resume_id}              → структурированное резюме (JSON, не PDF!)
  → Сохранить новых кандидатов в candidates
  → Запустить Scoring Pipeline на новых парах (vacancy_id × candidate_id)
```

**Важно:** HH.ru API возвращает резюме уже в структурированном JSON —
отдельные поля для опыта, образования, навыков. Парсинг PDF не нужен.

### Режим 2: Ручная загрузка (запасной)

```
Рекрутёр через Web UI
  → Загружает текст вакансии (textarea или PDF)
  → Загружает резюме (PDF/TXT файлы, мультизагрузка)
  → Нажимает "Оценить"
  → Scoring Pipeline запускается немедленно
```

---

## Схема базы данных (PostgreSQL)

### Таблица `vacancies`

```sql
vacancies (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  hh_vacancy_id  VARCHAR UNIQUE,       -- NULL для ручных вакансий
  title          VARCHAR NOT NULL,
  description    TEXT NOT NULL,        -- Исходный текст вакансии
  parsed         JSONB,                -- ParsedVacancy (LLM-извлечение)
  employer_name  VARCHAR,
  salary_from    INT,
  salary_to      INT,
  salary_currency VARCHAR,
  area           VARCHAR,
  status         VARCHAR DEFAULT 'active',  -- active | archived
  source         VARCHAR NOT NULL,     -- 'hh' | 'manual'
  last_synced_at TIMESTAMP,
  created_at     TIMESTAMP DEFAULT now()
)
```

`ParsedVacancy` (JSONB):
```json
{
  "required_skills": ["Python", "FastAPI", "PostgreSQL"],
  "nice_to_have": ["Kubernetes", "Redis"],
  "min_experience_years": 3,
  "level": "middle",
  "responsibilities": ["..."]
}
```

### Таблица `candidates`

```sql
candidates (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  hh_resume_id   VARCHAR UNIQUE,       -- NULL для ручных загрузок
  name           VARCHAR,              -- ФИО если доступно
  raw_text       TEXT,                 -- Исходный текст резюме
  parsed         JSONB,                -- ParsedResume (LLM-извлечение)
  source         VARCHAR NOT NULL,     -- 'hh' | 'manual'
  created_at     TIMESTAMP DEFAULT now(),
  updated_at     TIMESTAMP DEFAULT now()
)
```

`ParsedResume` (JSONB):
```json
{
  "skills": ["Python", "FastAPI", "asyncio"],
  "experience_years": 4,
  "education": "МГТУ, Прикладная математика",
  "last_position": "Backend Engineer @ Tinkoff",
  "summary": "..."
}
```

### Таблица `scoring_results`

```sql
scoring_results (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  vacancy_id    UUID REFERENCES vacancies(id),
  candidate_id  UUID REFERENCES candidates(id),
  score         INT NOT NULL CHECK (score BETWEEN 0 AND 100),
  grade         VARCHAR NOT NULL,      -- strong_match | good_match | weak_match | no_match
  key_matches   JSONB NOT NULL,        -- list[str], топ-3 преимущества
  key_gaps      JSONB NOT NULL,        -- list[str], топ-3 пробела ([] если нет)
  summary       TEXT NOT NULL,         -- Выжимка для рекрутёра
  confidence    FLOAT NOT NULL,        -- 0.0–1.0
  model_used    VARCHAR NOT NULL,      -- 'deepseek-chat' | 'claude-haiku-4-5' | ...
  created_at    TIMESTAMP DEFAULT now(),
  UNIQUE(vacancy_id, candidate_id)     -- один результат на пару
)
```

### Таблица `hh_sync_log` (история синхронизаций)

```sql
hh_sync_log (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  vacancy_id        UUID REFERENCES vacancies(id),
  candidates_fetched INT DEFAULT 0,
  candidates_new     INT DEFAULT 0,
  candidates_scored  INT DEFAULT 0,
  status            VARCHAR,           -- success | partial | failed
  error_message     TEXT,
  started_at        TIMESTAMP DEFAULT now(),
  finished_at       TIMESTAMP
)
```

---

## Обновлённый LLM-пайплайн

Обе сущности (вакансия и резюме) проходят LLM-парсинг один раз и сохраняются в БД.
Скоринг запускается на уже распарсенных данных.

```
Шаг 1: ParseVacancy (если vacancy.parsed IS NULL)
  Вход:  vacancy.description (текст)
  Вызов: LLM → ParsedVacancy JSON
  Сохр:  vacancies.parsed = result

Шаг 2: ParseResume (если candidate.parsed IS NULL)
  Вход:  candidate.raw_text
  Вызов: LLM → ParsedResume JSON
  Сохр:  candidates.parsed = result

Шаг 3: Score (если scoring_results WHERE vacancy_id + candidate_id не существует)
  Вход:  vacancy.parsed + candidate.parsed + anchor_candidate
  Вызов: LLM → CandidateScore JSON
  Сохр:  INSERT scoring_results

  Если scoring_results уже есть → пропустить (идемпотентность)
```

**Кэш = БД.** Повторный прогон одной вакансии с теми же кандидатами не делает
LLM-вызовов — результаты уже в `scoring_results`.

---

## Модели данных (Pydantic)

```python
class ParsedVacancy(BaseModel):
    required_skills: list[str]
    nice_to_have: list[str]
    min_experience_years: int
    level: str  # junior | middle | senior | lead
    responsibilities: list[str]

class ParsedResume(BaseModel):
    skills: list[str]
    experience_years: int
    education: str
    last_position: str
    summary: str

class CandidateScore(BaseModel):
    score: int  # 0–100
    grade: Literal["strong_match", "good_match", "weak_match", "no_match"]
    key_matches: list[str]  # топ-3, может быть пустым
    key_gaps: list[str]     # топ-3 пробела; [] если кандидат идеальный
    summary: str            # 2–3 предложения рекрутёру
    confidence: float       # 0.0–1.0
```

---

## LLM-провайдер: выбор и обоснование

**Проблема:** Прямой доступ к Anthropic API и OpenAI API из России затруднён
(платёжные ограничения, геоблокировка для части сервисов).

**Решение:** Абстрактный LLM-клиент с поддержкой нескольких провайдеров,
переключение через env var `LLM_PROVIDER`.

| Провайдер | Доступность из РФ | Качество | Цена | Function calling |
|---|---|---|---|---|
| **DeepSeek** (рекомендован) | Отличная | Высокое (DeepSeek-V3) | Очень низкая | Да |
| **OpenRouter** | Хорошая | Зависит от модели | Средняя | Да (проксирует Claude/GPT) |
| **GigaChat** | Отличная | Среднее для EN | Низкая | Ограниченно |
| **Anthropic (прямой)** | Нужен VPN/иностр. карта | Высокое | Средняя | Да (tool_use) |

**Выбор по умолчанию: DeepSeek** (`deepseek-chat`)
- Открытый API, доступен из РФ, принимает рублёвые карты через посредников
- Поддерживает OpenAI-совместимый function calling
- DeepSeek-V3: качество сопоставимо с Claude Sonnet при значительно меньшей цене
- ~$0.0003–0.001 за вызов (дешевле Claude Haiku)

**Архитектура клиента:**

```python
class LLMClient(Protocol):
    async def structured_call(self, messages, tool_schema) -> dict: ...

class DeepSeekClient(LLMClient):     # openai-compatible API
    ...

class OpenRouterClient(LLMClient):   # openai-compatible + ключ OpenRouter
    ...

class AnthropicClient(LLMClient):    # нативный Anthropic SDK (tool_use)
    ...
```

Переключение: `LLM_PROVIDER=deepseek|openrouter|anthropic` в `.env`

---

## HH.ru OAuth: ключевые аспекты

Для получения резюме откликнувшихся кандидатов нужна **employer OAuth**:

```
GET https://hh.ru/oauth/authorize?
    response_type=code&client_id=...&redirect_uri=...

→ Получить access_token (employer scope)
→ GET /vacancies?employer_id={id}              → список вакансий работодателя
→ GET /negotiations?vacancy_id={id}            → список откликов
→ GET /resumes/{resume_id}                     → структурированное резюме (JSON!)
```

**Важно: резюме через API — уже структурированный JSON**, не PDF.
Поля: `experience[]`, `education[]`, `skill_set[]`, `title` (последняя должность).
Парсинг PDF нужен только для режима ручной загрузки.

---

## Страницы Web UI

1. **Вакансии** — список активных вакансий из HH.ru, статус последней синхронизации
2. **Кандидаты по вакансии** — ранжированная таблица: ранг / имя / оценка / grade / матчи / пробелы / выжимка
3. **Ручная оценка** — форма загрузки (для вакансий не из HH.ru или тестирования)
4. **История синхронизаций** — лог hh_sync_log

---

## Защита от слабых мест LLM

| Проблема | Решение |
|---|---|
| Непоследовательные оценки | `temperature=0` + structured output (function calling) |
| Score drift при большом батче | Anchor-кандидат в каждом промпте Scoring |
| Пустые `key_gaps` у идеального кандидата | Явно в промпте: "если пробелов нет — верни пустой массив" |
| Галлюцинации | Pydantic validation + retry при parse error |
| Rate limit | Semaphore(5) + exponential backoff |

---

## Верификация (done = все пункты выполнены)

1. Scheduler синхронизировал вакансии: `vacancies` содержит записи с `hh_vacancy_id`
2. Кандидаты из откликов появились в `candidates` с заполненным `parsed`
3. `scoring_results` содержит оценки; повторный запуск не делает новых LLM-вызовов
4. Web UI показывает ранжированную таблицу по вакансии
5. У идеального кандидата `key_gaps = []`
6. `model_used` в scoring_results отражает текущий `LLM_PROVIDER` из `.env`
7. Ручная загрузка тоже работает: POST с PDF резюме → оценка
