"""Operate the remote Browser Gateway without authentication."""
import argparse
import json

from tools.browser.actions import BrowserAction
from tools.browser.remote import RemoteBrowserBackend
from tools.browser.session import BrowserSession


def main():
    parser = argparse.ArgumentParser(description="AGI remote browser client")
    parser.add_argument("--gateway", required=True, help="Gateway base URL, for example http://10.42.0.8:8080")
    parser.add_argument("--url", default=None, help="Open a URL before observing the browser")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    backend = RemoteBrowserBackend(args.gateway)
    print(json.dumps(backend.health(), ensure_ascii=False, indent=2))
    session = BrowserSession(backend, max_steps=args.steps)
    if args.url:
        result = session.execute(BrowserAction.open(args.url))
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(backend.observe().to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
