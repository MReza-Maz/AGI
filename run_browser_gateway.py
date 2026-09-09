"""Launch the network Browser Gateway on the browser host."""
import argparse
import json

from tools.browser.cdp import CDPBackend
from tools.browser.environment import BrowserEnvironment
from tools.browser.gateway import BrowserGateway
from tools.browser.session import BrowserSession


def main():
    parser = argparse.ArgumentParser(description="AGI remote browser gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cdp-host", default="127.0.0.1")
    parser.add_argument("--cdp-port", type=int, default=9222)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-timeout", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    backend = CDPBackend(args.cdp_host, args.cdp_port)
    session = BrowserSession(backend, max_steps=50)
    environment = BrowserEnvironment(session)
    BrowserGateway(
        environment,
        config=config,
        model_url=args.model_url,
        model=args.model,
        model_timeout=args.model_timeout,
        max_steps=args.max_steps,
    ).serve(args.host, args.port)


if __name__ == "__main__":
    main()
