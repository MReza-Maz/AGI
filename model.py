#!/usr/bin/env python3
"""Ollama client used to generate candidate source code."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class ModelError(RuntimeError):
    pass


class OllamaModel:
    def __init__(
        self,
        model="qwen3:4b",
        url="http://127.0.0.1:11434/api/generate",
        timeout=600,
        max_tokens=2048,
    ):
        self.model = model
        self.url = url
        self.timeout = int(timeout)
        self.max_tokens = int(max_tokens)

    def generate(self, source: str, instruction: str):
        prompt = f"""You are a software optimization engine.

Current Python program:
```python
{source}
```

Task: {instruction}

Rules:
- Return ONLY the complete replacement Python source code.
- Keep the program's externally observable purpose and output semantics.
- Do not modify files outside the requested program.
- Do not add network access, subprocesses, shell commands, or destructive behavior.
- Prefer measurable runtime improvements.
- The result must be valid standalone Python.
- The program must continue to support the --self-test argument.
"""
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": self.max_tokens,
                },
                "keep_alive": "10m",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise ModelError(f"Ollama request timed out after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            raise ModelError(f"Ollama connection failed: {exc.reason}") from exc
        except Exception as exc:
            raise ModelError(f"Ollama request failed: {exc}") from exc

        text = str(data.get("response", "")).strip()
        if not text:
            raise ModelError("model returned empty response")
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if text.lstrip().startswith("python"):
                    text = text.lstrip()[6:]
        return text.strip()
