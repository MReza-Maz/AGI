"""Run the browser environment against a local Chromium CDP session."""
import argparse
import json

from cognition.agent import CognitiveAgent
from tools.browser.cdp import CDPBackend
from tools.browser.environment import BrowserEnvironment
from tools.browser.session import BrowserSession


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="AGI browser environment")
    parser.add_argument("url", nargs="?", default="https://example.com")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9222)
    args = parser.parse_args()

    config = load_config(args.config)
    browser_cfg = config.get("browser", {})
    backend = CDPBackend(args.host, args.port, browser_cfg.get("timeout", 15))
    environment = BrowserEnvironment(BrowserSession(backend, browser_cfg.get("max_steps", 20)))
    agent = CognitiveAgent(config=config)
    agent.tools.register("browser.open", lambda a: environment.execute_dict({"name": "open", "args": a}),
                         "Open a URL in the browser")
    agent.tools.register("browser.click", lambda a: environment.execute_dict({"name": "click", "args": a}),
                         "Click a visible element")
    agent.tools.register("browser.type", lambda a: environment.execute_dict({"name": "type", "args": a}),
                         "Type into a visible input")
    agent.tools.register("browser.scroll", lambda a: environment.execute_dict({"name": "scroll", "args": a}),
                         "Scroll the page")
    agent.tools.register("browser.extract", lambda a: environment.execute_dict({"name": "extract", "args": a}),
                         "Extract visible page text")

    opened = agent.execute_tool("browser.open", {"url": args.url})
    print(json.dumps(opened, ensure_ascii=False, indent=2))
    observation = environment.observe()
    agent.perceive(observation.get("text", ""), {"url": observation.get("url"), "title": observation.get("title")})
    print(json.dumps(observation, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
