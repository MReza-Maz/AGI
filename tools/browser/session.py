"""Browser session abstraction with a stdlib HTTP backend and pluggable automation backend."""
import html
import re
import time
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .actions import BrowserAction
from .observation import BrowserObservation


class BrowserBackend:
    """Interface implemented by real browser automation backends."""
    def execute(self, action):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError


class HttpBrowserBackend(BrowserBackend):
    """Safe read-only fallback for web perception using Python's standard library."""
    def __init__(self, timeout=15, user_agent="AGI-Research-Agent/0.1"):
        self.timeout = int(timeout)
        self.user_agent = user_agent
        self.url = ""
        self.title = ""
        self.html = ""
        self.text = ""

    @staticmethod
    def _text(markup):
        markup = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", markup)
        markup = re.sub(r"(?s)<[^>]+>", " ", markup)
        return re.sub(r"\s+", " ", html.unescape(markup)).strip()

    def _open(self, url):
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read(2_000_000)
            charset = response.headers.get_content_charset() or "utf-8"
            self.html = raw.decode(charset, errors="replace")
            self.url = response.geturl()
        match = re.search(r"(?is)<title[^>]*>(.*?)</title>", self.html)
        self.title = self._text(match.group(1)) if match else ""
        self.text = self._text(self.html)

    def execute(self, action):
        name, args = action.name, action.args
        if name == "open":
            self._open(args["url"])
        elif name == "wait":
            time.sleep(max(0.0, min(float(args.get("seconds", 1)), 30.0)))
        elif name == "extract":
            return self.text
        else:
            raise NotImplementedError(
                f"HTTP backend is read-only; action '{name}' needs an interactive browser backend"
            )
        return self.observe()

    def observe(self):
        return BrowserObservation(url=self.url, title=self.title, text=self.text, dom=self.html)


class BrowserSession:
    """Validate and execute abstract actions against a selected backend."""
    ALLOWED = {"open", "click", "type", "scroll", "back", "wait", "extract"}

    def __init__(self, backend, max_steps=20):
        self.backend = backend
        self.max_steps = int(max_steps)
        self.steps = 0

    def observe(self):
        return self.backend.observe()

    def execute(self, action):
        if not isinstance(action, BrowserAction):
            raise TypeError("expected BrowserAction")
        if action.name not in self.ALLOWED:
            raise ValueError(f"unsupported browser action: {action.name}")
        if self.steps >= self.max_steps:
            raise RuntimeError("browser step budget exhausted")
        self.steps += 1
        return self.backend.execute(action)

    def reset_budget(self):
        self.steps = 0
