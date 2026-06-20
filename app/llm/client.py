from typing import Any, Protocol, runtime_checkable


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
