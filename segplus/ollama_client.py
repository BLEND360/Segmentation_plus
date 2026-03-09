"""Robust HTTP client for Ollama's local LLM API."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator, Optional

import requests

log = logging.getLogger("segplus.ollama_client")


class OllamaClient:
    """
    Production-grade client for Ollama's local LLM API.
    Features: health checks, exponential backoff retries, timeout,
    streaming support, structured JSON extraction.
    """

    def __init__(self, host: str, model: str, timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._endpoint = f"{self.host}/api/generate"

    def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            available = any(self.model in m for m in models)
            if not available:
                log.warning("Model '%s' not found. Available: %s", self.model, models)
            else:
                log.info("Ollama reachable. Model '%s' available.", self.model)
            return available
        except requests.RequestException as e:
            log.error("Ollama not reachable at %s: %s", self.host, e)
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response with retry logic. Returns full text."""
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        last_error = None
        for attempt in range(1, max_retries + 2):
            try:
                r = requests.post(
                    self._endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                response_text = r.json().get("response", "").strip()
                log.info("Ollama response received (%d chars)", len(response_text))
                return response_text

            except requests.Timeout:
                last_error = f"Timeout after {self.timeout}s"
            except requests.RequestException as e:
                last_error = str(e)

            if attempt <= max_retries:
                wait = 2 ** attempt
                log.warning(
                    "Ollama attempt %d failed (%s). Retrying in %ds...",
                    attempt, last_error, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"Ollama failed after {max_retries + 1} attempts: {last_error}")

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        """Streaming generation, yields tokens as they arrive."""
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        r = requests.post(
            self._endpoint,
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    break

    def extract_json(self, text: str) -> dict:
        """Extract the first valid JSON object from LLM output.
        Handles markdown fences, preamble text, and trailing text.
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"```", "", cleaned)

        # Try to find a JSON object
        match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Broader match: find outermost braces
        depth = 0
        start = None
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        start = None

        raise ValueError(f"No valid JSON found in LLM response:\n{text[:400]}")
