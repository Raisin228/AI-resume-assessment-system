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
