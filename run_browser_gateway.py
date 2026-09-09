"""Launch the network Browser Gateway on the browser host."""
import argparse

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
    args = parser.parse_args()

    backend = CDPBackend(args.cdp_host, args.cdp_port)
    session = BrowserSession(backend, max_steps=50)
    environment = BrowserEnvironment(session)
    BrowserGateway(environment).serve(args.host, args.port)


if __name__ == "__main__":
    main()
