"""First-class web environment for the cognitive agent."""
from .actions import BrowserAction
from .session import BrowserSession


class BrowserEnvironment:
    """Expose browser perception and actions through the common tool protocol."""
    def __init__(self, session):
        if not isinstance(session, BrowserSession):
            raise TypeError("session must be a BrowserSession")
        self.session = session

    def observe(self):
        return self.session.observe().to_dict()

    def execute(self, action):
        result = self.session.execute(action)
        return result.to_dict() if hasattr(result, "to_dict") else result

    def execute_dict(self, payload):
        if not isinstance(payload, dict) or not payload.get("name"):
            raise ValueError("browser action must contain a name")
        return self.execute(BrowserAction(payload["name"], dict(payload.get("args", {}))))
