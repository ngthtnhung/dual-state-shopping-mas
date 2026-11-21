from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    output_tokens: int
    elapsed_seconds: float


class OllamaClient:
    """Local-only Ollama text generation client using the standard library."""

    def __init__(
        self,
        model: str = "llama3:latest",
        host: str = "http://localhost:11434",
        timeout_seconds: float = 180.0,
    ) -> None:
        parsed = urlparse(host)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Ollama host must be an HTTP(S) URL")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama host must resolve to this machine")
        if not model.strip():
            raise ValueError("Ollama model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.endpoint = host.rstrip("/") + "/api/generate"
        self.timeout_seconds = timeout_seconds

    def generate(
        self, prompt: str, *, system: str, format_json: bool = False
    ) -> Generation:
        body = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
        }
        if format_json:
            body["format"] = "json"
        payload = json.dumps(body).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Ollama generation failed for model {self.model!r}: {error}"
            ) from error
        elapsed = time.perf_counter() - started
        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                f"Ollama returned no generated text for model {self.model!r}"
            )
        return Generation(
            text=text.strip(),
            prompt_tokens=int(body.get("prompt_eval_count", 0)),
            output_tokens=int(body.get("eval_count", 0)),
            elapsed_seconds=elapsed,
        )
