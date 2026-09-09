import unittest

from tools.registry import ToolRegistry
from tools.browser.actions import BrowserAction
from tools.browser.observation import BrowserObservation
from tools.browser.session import BrowserBackend, BrowserSession
from tools.browser.environment import BrowserEnvironment


class FakeBackend(BrowserBackend):
    def __init__(self):
        self.url = ""
        self.calls = []

    def execute(self, action):
        self.calls.append(action.to_dict())
        if action.name == "open":
            self.url = action.args["url"]
        return self.observe()

    def observe(self):
        return BrowserObservation(url=self.url, title="fake", text="hello world")


class ToolTests(unittest.TestCase):
    def test_registry_blocks_dangerous_tool(self):
        registry = ToolRegistry(require_approval=True)
        registry.register("danger", lambda _: "ok", dangerous=True)
        result = registry.execute("danger")
        self.assertTrue(result["blocked"])
        self.assertFalse(result["executed"])
        self.assertEqual(registry.execute("danger", approved=True)["result"], "ok")

    def test_browser_session_protocol(self):
        backend = FakeBackend()
        env = BrowserEnvironment(BrowserSession(backend, max_steps=3))
        result = env.execute(BrowserAction.open("https://example.com"))
        self.assertEqual(result["url"], "https://example.com")
        self.assertEqual(env.observe()["text"], "hello world")
        self.assertEqual(len(backend.calls), 1)


if __name__ == "__main__":
    unittest.main()
