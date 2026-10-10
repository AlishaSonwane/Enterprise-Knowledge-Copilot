"""Groq opt-in transport. One call, fixed HTTPS destination, no fallbacks.
The account must stay on Free. The API does not attest account billing tier.
"""
import os
import httpx
import config

class CloudGenerationError(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)

async def generate_cloud_answer(messages):
    if os.environ.get("GROQ_FREE_PLAN_CONFIRMED") != "1":
        raise CloudGenerationError("cloud_free_plan_unconfirmed")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise CloudGenerationError("cloud_key_missing")
    payload = {"model": config.GROQ_MODEL, "messages": messages,
               "temperature": config.TEMPERATURE, "stream": False,
               "max_completion_tokens": config.GROQ_MAX_COMPLETION_TOKENS,
               "reasoning_effort": "low"}
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=False,
                                     trust_env=False) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": "Bearer " + key}, json=payload)
    except httpx.TimeoutException:
        raise CloudGenerationError("cloud_timeout") from None
    except httpx.RequestError:
        raise CloudGenerationError("cloud_unavailable") from None
    status = response.status_code
    if status == 429:
        raise CloudGenerationError("cloud_rate_limited")
    if status in (401,403):
        raise CloudGenerationError("cloud_auth_error")
    if status != 200:
        raise CloudGenerationError("cloud_http_error")
    try:
        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") != "stop":
            raise CloudGenerationError("cloud_incomplete_answer")
        if not isinstance(content, str) or not content.strip():
            raise CloudGenerationError("cloud_empty_answer")
        return content
    except (ValueError,KeyError,IndexError,TypeError):
        raise CloudGenerationError("cloud_invalid_response") from None
