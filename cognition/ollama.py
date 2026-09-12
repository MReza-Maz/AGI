"""Minimal standard-library client for a local Ollama reasoning model."""

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


class OllamaReasoner:
    """Use a local Ollama model as the language and reasoning component."""

    def __init__(self, model="qwen3:30b", endpoint="http://127.0.0.1:11434/api/chat", timeout=120):
        self.model = str(model)
        self.endpoint = str(endpoint)
        self.timeout = max(1, int(timeout))

    def chat(self, messages, temperature=0.2, json_mode=False):
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        if json_mode:
            payload["format"] = "json"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {error.code}: {body}") from error
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise OllamaError(f"cannot reach Ollama at {self.endpoint}: {error}") from error

        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise OllamaError("invalid Ollama response")
        content = data["message"].get("content", "")
        if not isinstance(content, str):
            raise OllamaError("Ollama returned non-text content")
        return content

    def reason(self, system, context, temperature=0.2):
        """Return a structured decision for the AGI runtime."""
        schema = (
            "Return JSON only with keys: understanding, assumptions, hypotheses, "
            "plan, selected_action, confidence, uncertainty, final_response. "
            "hypotheses and plan must be arrays. selected_action must be an object "
            "with keys tool, arguments, reason, expected_outcome, risk; use null "
            "for tool when no external action is needed."
        )
        content = self.chat(
            [
                {"role": "system", "content": system + "\n" + schema},
                {"role": "user", "content": context},
            ],
            temperature=temperature,
            json_mode=True,
        )
        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            raise OllamaError(f"reasoning response is not valid JSON: {content}") from error
        if not isinstance(result, dict):
            raise OllamaError("reasoning response must be a JSON object")
        return result
