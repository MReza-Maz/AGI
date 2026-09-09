"""Test and operate a remote Browser Gateway from the AGI host."""
import argparse
import json

from tools.browser.actions import BrowserAction
from tools.browser.remote import RemoteBrowserBackend
from tools.browser.session import BrowserSession


def main():
    parser = argparse.ArgumentParser(description="AGI remote browser client")
    parser.add_argument("--gateway", required=True, help="Gateway base URL, for example http://192.168.1.50:8080")
    parser.add_argument("--token", required=True)
    parser.add_argument("--url", default=None)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    backend = RemoteBrowserBackend(args.gateway, token=args.token)
    print(json.dumps(backend.health(), ensure_ascii=False, indent=2))
    if args.url:
        result = BrowserSession(backend, max_steps=args.steps).execute(BrowserAction.open(args.url))
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(backend.observe().to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
