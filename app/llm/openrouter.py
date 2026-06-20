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
