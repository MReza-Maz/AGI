import unittest

from cognition.runtime import CognitiveRuntime


class FakeEnvironment:
    def __init__(self):
        self.observations = [{"url": "", "title": "", "text": ""}]
        self.actions = []

    def observe(self):
        return dict(self.observations[-1])

    def execute_dict(self, action):
        self.actions.append(action)
        if action["name"] == "open":
            self.observations.append({
                "url": action["args"]["url"],
                "title": "Example",
                "text": "Example page",
            })
        return {"ok": True, "action": action}


class TestRuntime(unittest.TestCase):
    def test_final_response(self):
        runtime = CognitiveRuntime(FakeEnvironment())
        runtime._ask_model = lambda messages: {"type": "final", "response": "سلام!"}
        result = runtime.handle("سلام")
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "cognitive")
        self.assertEqual(result["response"], "سلام!")
        self.assertEqual(result["steps"], [])

    def test_browser_action_then_final(self):
        environment = FakeEnvironment()
        runtime = CognitiveRuntime(environment, max_steps=3)
        decisions = iter([
            {"type": "action", "thought": "Open the requested site.", "action": {"name": "open", "args": {"url": "https://example.com"}}},
            {"type": "final", "response": "The page is open."},
        ])
        runtime._ask_model = lambda messages: next(decisions)
        result = runtime.handle("open example.com")
        self.assertEqual(environment.actions[0]["name"], "open")
        self.assertEqual(result["response"], "The page is open.")
        self.assertEqual(len(result["steps"]), 1)

    def test_invalid_action_is_rejected(self):
        runtime = CognitiveRuntime(FakeEnvironment())
        with self.assertRaises(ValueError):
            runtime._validate_action({"name": "shell", "args": {}})


if __name__ == "__main__":
    unittest.main()
