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
