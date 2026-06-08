import pytest
import respx
from httpx import Response

from llm_rag.providers.ollama import OllamaProvider


@pytest.mark.mocked
@respx.mock
async def test_provider_returns_response():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(200, json={"response": "hello"})
    )
    out = await OllamaProvider().generate("hi")
    assert out == "hello"
