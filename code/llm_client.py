"""
LLM client supporting OpenAI, Ollama (Qwen/local), and any OpenAI-compatible endpoint.

Quick start options (all free):
  Ollama (local):   ollama pull qwen2.5-coder:7b  →  set BACKEND=ollama
  Together AI:      set OPENAI_API_KEY=<together-key> BACKEND=together
  OpenAI:           set OPENAI_API_KEY=<openai-key>   BACKEND=openai  (default)
"""
import time
import re
import config

# Base URLs for supported backends
_BACKENDS = {
    "openai":   ("https://api.openai.com/v1",        config.OPENAI_API_KEY),
    "ollama":   ("http://localhost:11434/v1",         "ollama"),   # Ollama needs any non-empty key
    "together": ("https://api.together.xyz/v1",       config.OPENAI_API_KEY),
    "groq":     ("https://api.groq.com/openai/v1",    config.OPENAI_API_KEY),
}

# Recommended free Qwen models per backend
QWEN_MODELS = {
    "ollama":   "qwen2.5-coder:7b",      # pull with: ollama pull qwen2.5-coder:7b
    "together": "Qwen/Qwen2.5-Coder-7B-Instruct-Turbo",
    "groq":     "qwen-qwq-32b",
    "openai":   config.MODEL,            # gpt-4o-mini
}


class LLMClient:
    def __init__(self, model: str = None, api_key: str = None, backend: str = None):
        self.backend = backend or config.BACKEND
        base_url, default_key = _BACKENDS.get(self.backend, _BACKENDS["openai"])
        self.base_url = base_url
        self.api_key = api_key or default_key or "none"
        self.model = model or QWEN_MODELS.get(self.backend, config.MODEL)
        self._total_tokens = 0
        self._total_calls = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    def call(
        self,
        prompt: str,
        system: str = "You are an expert Python programmer and debugger.",
        max_tokens: int = None,
        temperature: float = None,
    ) -> dict:
        max_tokens = max_tokens or config.MAX_TOKENS_HYPOTHESIS
        temperature = temperature if temperature is not None else config.TEMPERATURE

        for attempt in range(config.MAX_RETRIES + 1):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""
                # Ollama may not report token usage
                tokens = response.usage.total_tokens if response.usage else len(prompt.split()) * 2
                self._total_tokens += tokens
                self._total_calls += 1
                return {"content": content, "tokens": tokens}
            except Exception as e:
                if attempt < config.MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"LLM call failed after {config.MAX_RETRIES} retries: {e}")

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_calls(self) -> int:
        return self._total_calls

    def reset_counters(self):
        self._total_tokens = 0
        self._total_calls = 0


class MockLLMClient(LLMClient):
    """Deterministic mock client for testing without an API key."""

    def __init__(self, model: str = "mock"):
        self.model = model
        self._total_tokens = 0
        self._total_calls = 0
        self._client = None

    def call(self, prompt: str, system: str = "", max_tokens: int = None, temperature: float = None) -> dict:
        self._total_calls += 1
        tokens = len(prompt.split()) * 2

        if "hypothesis" in prompt.lower() or "hypothes" in prompt.lower():
            content = (
                "HYPOTHESIS 1: Off-by-one error in loop bounds\n"
                "EXPLANATION: The loop iterates one step too far or stops one step early.\n"
                "LOCATION: for loop range\n\n"
                "HYPOTHESIS 2: Wrong comparison operator\n"
                "EXPLANATION: A <= should be < or vice versa in a conditional.\n"
                "LOCATION: if statement condition\n\n"
                "HYPOTHESIS 3: Incorrect return value\n"
                "EXPLANATION: The function returns a wrong variable or expression.\n"
                "LOCATION: return statement\n"
            )
        elif "fix" in prompt.lower() or "FIX" in prompt:
            # Extract the buggy code from the prompt and try to pass through
            lines = prompt.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if "```python" in line:
                    in_code = True
                    continue
                if "```" in line and in_code:
                    in_code = False
                    continue
                if in_code:
                    code_lines.append(line)
            buggy = "\n".join(code_lines) if code_lines else "def f(): return None"
            content = (
                f"FIX 1:\n```python\n{buggy}\n```\n\n"
                f"FIX 2:\n```python\n{buggy}\n```\n\n"
                f"FIX 3:\n```python\n{buggy}\n```\n"
            )
        elif any(c.isdigit() for c in prompt[-30:]):
            content = "7"
        else:
            content = "5"

        self._total_tokens += tokens
        return {"content": content, "tokens": tokens}
