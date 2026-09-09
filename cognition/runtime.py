"""Model-backed cognitive runtime for conversational and browser tasks.

The runtime uses a local Ollama-compatible HTTP endpoint and keeps browser actions
inside the existing BrowserEnvironment boundary. It is an AGI research component,
not a claim of artificial general intelligence.
"""
import json
import os
import urllib.error
import urllib.request

from cognition.agent import CognitiveAgent


class CognitiveRuntime:
    """Coordinate language reasoning, memory, and iterative browser interaction."""

    ACTIONS = {"open", "click", "type", "scroll", "back", "wait", "extract"}

    def __init__(self, environment, config=None, model_url=None, model=None,
                 timeout=120, max_steps=8):
        self.environment = environment
        self.config = config or {}
        self.model_url = model_url or os.environ.get(
            "AGI_CHAT_MODEL_URL", "http://127.0.0.1:11434/api/chat"
        )
        self.model = model or os.environ.get("AGI_CHAT_MODEL", "qwen2.5-coder:7b")
        self.timeout = int(timeout)
        self.max_steps = max(1, int(max_steps))
        self.cognitive = CognitiveAgent(config=self.config, inference=None)
        self.conversation = []

    @staticmethod
    def _clip(value, limit):
        text = str(value or "")
        return text if len(text) <= limit else text[:limit] + "..."

    def _observation_text(self, observation):
        if not isinstance(observation, dict):
            return str(observation)
        return json.dumps({
            "url": observation.get("url", ""),
            "title": observation.get("title", ""),
            "text": self._clip(observation.get("text", ""), 9000),
            "dom": self._clip(observation.get("dom", ""), 4000),
            "accessibility": self._clip(observation.get("accessibility", ""), 3000),
        }, ensure_ascii=False)

    def _system_prompt(self):
        return """You are the cognitive core of an AGI research system.
Answer the user directly when no external action is needed. You can operate the
browser through ONLY the abstract actions listed below. Never invent tools or raw
browser code.

Available browser actions:
- open: {url}
- click: {target} (visible text, label, id, or selector)
- type: {target, text}
- scroll: {amount}
- back: {}
- wait: {seconds}
- extract: {target}

Return ONLY valid JSON in exactly one of these forms:
{"type":"final","response":"your answer"}
{"type":"action","thought":"short reason","action":{"name":"open|click|type|scroll|back|wait|extract","args":{...}}}

For web research, perform actions iteratively, inspect observations, and continue
until there is enough evidence to answer. Do not claim to have browsed if you did
not receive a browser observation. Prefer concise, useful final answers.
"""

    def _messages(self, prompt, observation):
        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.conversation[-8:])
        memory = self.cognitive.recall(prompt, k=3)
        memory_text = "\n".join(item.get("text", "") for item in memory)
        context = (
            "Current browser observation:\n" + self._observation_text(observation)
            + "\n\nRelevant memory:\n" + self._clip(memory_text, 2500)
        )
        messages.append({"role": "system", "content": context})
        return messages

    def _ask_model(self, messages):
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }).encode("utf-8")
        request = urllib.request.Request(
            self.model_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"model HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"cannot reach local language model at {self.model_url}: {error.reason}"
            ) from error
        text = data.get("message", {}).get("content", data.get("response", ""))
        if not text:
            raise RuntimeError("language model returned an empty response")
        try:
            result = json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError("language model returned invalid JSON") from error
        if not isinstance(result, dict):
            raise RuntimeError("language model response must be a JSON object")
        return result

    def _validate_action(self, value):
        if not isinstance(value, dict):
            raise ValueError("model action must be an object")
        name = value.get("name")
        args = value.get("args", {})
        if name not in self.ACTIONS:
            raise ValueError(f"unsupported browser action: {name}")
        if not isinstance(args, dict):
            raise ValueError("browser action args must be an object")
        return {"name": name, "args": args}

    def handle(self, prompt):
        """Handle one user turn, including iterative browser tool use when needed."""
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("prompt is required")

        observation = self.environment.observe()
        self.cognitive.think(prompt, k=3)
        self.conversation.append({"role": "user", "content": prompt})
        steps = []

        try:
            for index in range(self.max_steps):
                decision = self._ask_model(self._messages(prompt, observation))
                kind = decision.get("type")
                if kind == "final":
                    response = str(decision.get("response", "")).strip()
                    if not response:
                        raise RuntimeError("language model returned an empty final response")
                    self.conversation.append({"role": "assistant", "content": response})
                    self.cognitive.perceive(response, {"type": "assistant_response", "turn": self.cognitive.turn})
                    return {
                        "ok": True,
                        "mode": "cognitive",
                        "response": response,
                        "steps": steps,
                        "observation": observation,
                    }

                if kind != "action":
                    raise ValueError("model decision type must be 'final' or 'action'")

                action = self._validate_action(decision.get("action"))
                result = self.environment.execute_dict(action)
                steps.append({
                    "index": index + 1,
                    "thought": self._clip(decision.get("thought", ""), 500),
                    "action": action,
                    "result": result,
                })
                observation = self.environment.observe()
                self.cognitive.perceive(
                    self._observation_text(observation),
                    {"type": "browser_observation", "turn": self.cognitive.turn},
                )
                if isinstance(result, dict) and result.get("error"):
                    break

            # Ask the model for a final answer after the action budget is exhausted.
            messages = self._messages(prompt, observation)
            messages.append({
                "role": "system",
                "content": "The browser action budget is exhausted. Give the best evidence-based final answer now.",
            })
            decision = self._ask_model(messages)
            response = str(decision.get("response", "")).strip()
            if not response:
                raise RuntimeError("model did not provide a final answer")
            self.conversation.append({"role": "assistant", "content": response})
            return {
                "ok": True,
                "mode": "cognitive",
                "response": response,
                "steps": steps,
                "observation": observation,
            }
        except Exception:
            # Keep the user turn in memory, but do not fabricate a successful answer.
            raise
