from app.core.config import Settings


def test_settings_load_defaults():
    s = Settings()
    assert s.LLM_PROVIDER == "deepseek"
    assert s.LLM_MODEL == "deepseek-chat"
    assert s.SYNC_INTERVAL_HOURS == 6


def test_settings_override_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    s = Settings()
    assert s.LLM_PROVIDER == "openrouter"
    assert s.LLM_MODEL == "gpt-4o-mini"
