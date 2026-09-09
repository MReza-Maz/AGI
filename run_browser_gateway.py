"""Launch the network Browser Gateway on the browser host."""
import argparse
import secrets

from tools.browser.cdp import CDPBackend
from tools.browser.environment import BrowserEnvironment
from tools.browser.gateway import BrowserGateway
from tools.browser.session import BrowserSession


def main():
    parser = argparse.ArgumentParser(description="AGI remote Browser Gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cdp-host", default="127.0.0.1")
    parser.add_argument("--cdp-port", type=int, default=9222)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    token = args.token or secrets.token_urlsafe(32)
    print("Gateway token:", token)
    backend = CDPBackend(args.cdp_host, args.cdp_port)
    session = BrowserSession(backend, max_steps=50)
    environment = BrowserEnvironment(session)
    BrowserGateway(environment, token=token).serve(args.host, args.port)


if __name__ == "__main__":
    main()
