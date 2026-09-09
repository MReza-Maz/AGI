"""Structured browser observations for perception and memory."""
from dataclasses import dataclass, field


@dataclass
class BrowserObservation:
    url: str = ""
    title: str = ""
    text: str = ""
    dom: str = ""
    accessibility: str = ""
    screenshot: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "dom": self.dom,
            "accessibility": self.accessibility,
            "screenshot": self.screenshot,
            "metadata": dict(self.metadata),
        }

    def summary(self, limit=12000):
        chunks = [f"URL: {self.url}", f"Title: {self.title}"]
        if self.text:
            chunks.append("Text:\n" + self.text)
        if self.accessibility:
            chunks.append("Accessibility:\n" + self.accessibility)
        return "\n\n".join(chunks)[:int(limit)]
