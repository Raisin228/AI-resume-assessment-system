from app.core.config import settings
from app.llm.client import LLMClient


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
