import pytest

from llm_rag.providers.ollama import OllamaProvider


@pytest.mark.ollama
async def test_live_ollama_responds():
    out = await OllamaProvider().generate("Say exactly the word: ok")
    assert isinstance(out, str) and len(out) > 0
