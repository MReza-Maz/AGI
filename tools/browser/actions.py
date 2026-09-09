"""Abstract browser actions. The cognitive core never emits raw browser code."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BrowserAction:
    name: str
    args: dict = field(default_factory=dict)

    def to_dict(self):
        return {"name": self.name, "args": dict(self.args)}

    @classmethod
    def open(cls, url):
        return cls("open", {"url": str(url)})

    @classmethod
    def click(cls, target):
        return cls("click", {"target": str(target)})

    @classmethod
    def type(cls, target, text):
        return cls("type", {"target": str(target), "text": str(text)})

    @classmethod
    def scroll(cls, amount=600):
        return cls("scroll", {"amount": int(amount)})

    @classmethod
    def back(cls):
        return cls("back")

    @classmethod
    def wait(cls, seconds=1):
        return cls("wait", {"seconds": float(seconds)})

    @classmethod
    def extract(cls, target="body"):
        return cls("extract", {"target": str(target)})
