# llm_client.py
# ─────────────────────────────────────────────────────────
# Thin wrapper around the Groq Chat Completions API.
# All LLM calls in the pipeline go through here.
# ─────────────────────────────────────────────────────────

import json
import requests

from config import GROQ_MODEL_LARGE

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_groq(api_key, messages, model=None, max_tokens=1800, temperature=0.1):
    """
    Send a chat completion request to Groq and return the assistant message text.

    Parameters
    ----------
    api_key     : Groq API key
    messages    : list of {"role": ..., "content": ...} dicts
    model       : Groq model ID (defaults to GROQ_MODEL_LARGE from config)
    max_tokens  : maximum tokens in the response
    temperature : sampling temperature (keep low for deterministic code generation)

    Returns
    -------
    str : the assistant's reply text

    Raises
    ------
    RuntimeError if the HTTP response is not 200.
    """
    model = model or GROQ_MODEL_LARGE
    headers = {
        "Content-Type" : "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = json.dumps({
        "model"      : model,
        "max_tokens" : max_tokens,
        "messages"   : messages,
        "temperature": temperature,
    })

    resp = requests.post(GROQ_API_URL, headers=headers, data=payload, timeout=90)

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:400])
        except Exception:
            detail = resp.text[:400]
        raise RuntimeError(f"Groq HTTP {resp.status_code}: {detail}")

    return resp.json()["choices"][0]["message"]["content"]
