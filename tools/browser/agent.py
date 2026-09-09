"""Prompt-driven browser agent for the remote AGI web interface.

The agent converts a small, deterministic subset of natural-language prompts into
abstract BrowserAction objects. It is intentionally model-independent so the web UI
works before a capable foundation model is connected to the cognitive core.
"""
import re
from urllib.parse import quote_plus

from .actions import BrowserAction


class BrowserPromptAgent:
    """Execute safe, high-level browser tasks and return observations."""

    URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
    DOMAIN_RE = re.compile(r"(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?", re.IGNORECASE)

    def __init__(self, environment, max_actions=8):
        self.environment = environment
        self.max_actions = max(1, int(max_actions))
        self.history = []

    @staticmethod
    def _clean_url(value):
        return value.rstrip(".,;:!?)]}")

    def _url_from_prompt(self, prompt):
        match = self.URL_RE.search(prompt)
        if match:
            return self._clean_url(match.group(0))
        match = self.DOMAIN_RE.search(prompt)
        if match:
            return "https://" + self._clean_url(match.group(0))
        return None

    @staticmethod
    def _search_query(prompt):
        patterns = (
            r"(?:search|google|bing|find|look up|research)\s+(?:for\s+)?(.+)",
            r"(?:جستجو|جستجو کن|پیدا کن|تحقیق کن|درباره)\s*(.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                query = match.group(1).strip(" .،")
                query = re.sub(r"\s+(?:on|in)\s+(?:the\s+)?web$", "", query, flags=re.IGNORECASE)
                if query:
                    return query
        return None

    @staticmethod
    def _target_after_keyword(prompt, keyword):
        match = re.search(rf"{keyword}\s+['\"]?([^'\"]+)['\"]?", prompt, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def plan(self, prompt):
        """Build abstract browser actions without emitting executable browser code."""
        text = str(prompt).strip()
        if not text:
            raise ValueError("prompt is required")

        actions = []
        url = self._url_from_prompt(text)
        if url:
            actions.append(BrowserAction.open(url))
        else:
            query = self._search_query(text)
            if query:
                actions.append(BrowserAction.open("https://www.google.com/search?q=" + quote_plus(query)))

        if re.search(r"\bback\b|\bgo back\b|\bبازگشت\b", text, re.IGNORECASE):
            actions.append(BrowserAction.back())

        if re.search(r"\bscroll\b|\bاسکرول\b|\bپایین\b", text, re.IGNORECASE):
            amount = 1000 if re.search(r"down|پایین", text, re.IGNORECASE) else -800
            actions.append(BrowserAction.scroll(amount))

        click_target = self._target_after_keyword(text, r"click(?:\s+on)?")
        if click_target:
            actions.append(BrowserAction.click(click_target))

        if not actions:
            actions.append(BrowserAction.wait(0.2))
        return actions[: self.max_actions]

    def run(self, prompt):
        actions = self.plan(prompt)
        steps = []
        for action in actions:
            result = self.environment.execute(action)
            step = {"action": action.to_dict(), "result": result}
            steps.append(step)
            if isinstance(result, dict) and result.get("error"):
                break
        observation = self.environment.observe()
        record = {"prompt": str(prompt), "steps": steps, "observation": observation}
        self.history.append(record)
        return record
