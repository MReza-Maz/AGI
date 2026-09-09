"""Browser environment exposed to the cognitive agent."""
from .actions import BrowserAction
from .session import BrowserSession

__all__ = ["BrowserAction", "BrowserSession"]
