"""Operate the remote Browser Gateway without authentication."""
import argparse
import json
import urllib.error
import urllib.request

from tools.browser.actions import BrowserAction
from tools.browser.remote import RemoteBrowserBackend
from tools.browser.session import BrowserSession


def post_prompt(gateway, prompt):
    """Send a prompt to the gateway prompt API using only the standard library."""
    url = gateway.rstrip("/") + "/api/prompt"
    body = json.dumps({"prompt": prompt}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"gateway returned HTTP {error.code}: {payload}") from error


def main():
    parser = argparse.ArgumentParser(description="AGI remote browser client")
    parser.add_argument("--gateway", required=True, help="Gateway base URL, for example http://10.42.0.8:8080")
    parser.add_argument("--url", default=None, help="Open a URL before observing the browser")
    parser.add_argument("--prompt", default=None, help="Send a natural-language prompt to the browser agent")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    if args.prompt:
        print(json.dumps(post_prompt(args.gateway, args.prompt), ensure_ascii=False, indent=2))
        return

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
