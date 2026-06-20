import pytest
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.models.base import Base
from app.models.vacancy import Vacancy
from app.models.candidate import Candidate
from app.models.scoring_result import ScoringResult
from app.models.sync_log import SyncLog  # noqa: F401 — needed for drop_all FK ordering

TEST_DB = "postgresql+asyncpg://postgres:postgres@localhost:5433/resume_scoring"


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


@pytest.mark.asyncio(loop_scope="module")
async def test_vacancy_create(db_session):
    v = Vacancy(title="Python Dev", description="FastAPI required", source="manual")
    db_session.add(v)
    await db_session.commit()
    assert v.id is not None
    assert v.status == "active"


@pytest.mark.asyncio(loop_scope="module")
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
