import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
