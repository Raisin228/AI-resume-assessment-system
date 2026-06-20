FROM python:3.12-slim
WORKDIR /app
RUN pip install poetry==2.2.1
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --no-root
COPY . .
