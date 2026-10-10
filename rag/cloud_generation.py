"""Groq opt-in transport. One call, fixed HTTPS destination, no fallbacks.
The account must stay on Free. The API does not attest account billing tier.
"""
import os
import json
import re
import httpx
import config

class CloudGenerationError(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)

STRUCTURED_INSTRUCTION = '''
Cloud output format override ONLY: return a JSON object with exactly one key "claims".
"claims" must contain 1 to 6 objects, each with exactly "text" and "source" string fields.
Each text is one concise factual sentence supported by the supplied context.
Each source is the exact context filename, without brackets or a citation marker.
Do not include citations inside text, headings, markdown fences or additional prose.
The application renders citations. All original evidence-only and uncertainty rules still apply.
Do not claim guaranteed correctness or elimination of hallucinations.
'''

def render_structured_answer(content):
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or set(parsed) != {'claims'}:
            raise ValueError()
        claims = parsed['claims']
        if not isinstance(claims, list) or not 1 <= len(claims) <= 6:
            raise ValueError()
        lines = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {'text','source'}:
                raise ValueError()
            text, source = claim['text'], claim['source']
            if (not isinstance(text,str) or not text.strip() or len(text)>1200
                or not isinstance(source,str) or not source.strip() or len(source)>200
                or re.search(r'[\r\n\[\]]',text+source)):
                raise ValueError()
            # Attribution is supplied by the model, then checked against retrieved
            # source filenames by generation.py. Never substitute a missing source.
            lines.append(text.strip()+' [Source: '+source.strip()+']')
        return '\n'.join(lines)
    except (ValueError,TypeError):
        raise CloudGenerationError('cloud_invalid_structure') from None

async def generate_cloud_answer(messages):
    if os.environ.get("GROQ_FREE_PLAN_CONFIRMED") != "1":
        raise CloudGenerationError("cloud_free_plan_unconfirmed")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise CloudGenerationError("cloud_key_missing")
    structured_messages = [dict(message) for message in messages]
    if structured_messages and structured_messages[0].get('role') == 'system':
        structured_messages[0]['content'] += '\n'+STRUCTURED_INSTRUCTION
    else:
        structured_messages.insert(0, {'role':'system','content':STRUCTURED_INSTRUCTION})
    payload = {"model": config.GROQ_MODEL, "messages": structured_messages,
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
        return render_structured_answer(content)
    except (ValueError,KeyError,IndexError,TypeError):
        raise CloudGenerationError("cloud_invalid_response") from None
