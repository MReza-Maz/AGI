import unittest

from tools.browser.agent import BrowserPromptAgent
from tools.browser.actions import BrowserAction
from tools.browser.environment import BrowserEnvironment
from tools.browser.observation import BrowserObservation
from tools.browser.session import BrowserBackend, BrowserSession


class FakeBackend(BrowserBackend):
    def __init__(self):
        self.url = "chrome://newtab/"
        self.calls = []

    def execute(self, action):
        self.calls.append(action.to_dict())
        if action.name == "open":
            self.url = action.args["url"]
        return self.observe()

    def observe(self):
        return BrowserObservation(url=self.url, title="fake", text="hello world")


class BrowserPromptAgentTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.environment = BrowserEnvironment(BrowserSession(self.backend, max_steps=10))
        self.agent = BrowserPromptAgent(self.environment)

    def test_url_prompt_opens_page(self):
        result = self.agent.run("Open https://example.com")
        self.assertEqual(result["observation"]["url"], "https://example.com")
        self.assertEqual(self.backend.calls[0]["name"], "open")

    def test_search_prompt_creates_search_url(self):
        result = self.agent.run("search for Docker architecture")
        self.assertIn("google.com/search?q=", result["observation"]["url"])

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            self.agent.run("   ")


if __name__ == "__main__":
    unittest.main()
