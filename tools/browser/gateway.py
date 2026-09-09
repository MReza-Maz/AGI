"""Network gateway that exposes the browser environment as a small JSON HTTP API.

Run this service on the machine that owns Chromium. Authentication is intentionally
omitted because this gateway is designed for a trusted/private network.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .actions import BrowserAction
from .environment import BrowserEnvironment


class BrowserGateway:
    """HTTP JSON gateway around a BrowserEnvironment."""
    def __init__(self, environment, max_body=64 * 1024):
        self.environment = environment
        self.max_body = int(max_body)

    def handler_class(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AGI-Browser-Gateway/1.1"

            def _json(self, status, payload):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    raise ValueError("invalid Content-Length")
                if length < 0 or length > gateway.max_body:
                    raise ValueError("request body too large")
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("JSON body must be an object")
                return value

            def do_GET(self):
                path = urlparse(self.path).path
                try:
                    if path == "/health":
                        return self._json(200, {"ok": True, "service": "browser-gateway", "auth": False})
                    if path == "/observe":
                        return self._json(200, {"ok": True, "observation": gateway.environment.observe()})
                    if path == "/tools":
                        return self._json(200, {"ok": True, "actions": ["open", "click", "type", "scroll", "back", "wait", "extract"]})
                    return self._json(404, {"error": "not found"})
                except Exception as exc:
                    return self._json(500, {"error": str(exc)})

            def do_POST(self):
                path = urlparse(self.path).path
                try:
                    payload = self._body()
                    if path == "/action":
                        name = payload.get("name")
                        if not isinstance(name, str):
                            raise ValueError("action name is required")
                        action = BrowserAction(name, dict(payload.get("args", {})))
                        return self._json(200, {"ok": True, "result": gateway.environment.execute(action)})
                    return self._json(404, {"error": "not found"})
                except (ValueError, TypeError, KeyError) as exc:
                    return self._json(400, {"error": str(exc)})
                except Exception as exc:
                    return self._json(500, {"error": str(exc)})

            def log_message(self, fmt, *args):
                return

        return Handler

    def serve(self, host="0.0.0.0", port=8080):
        server = ThreadingHTTPServer((host, int(port)), self.handler_class())
        server.daemon_threads = True
        print(f"Browser Gateway listening on {host}:{int(port)}")
        print("Authentication disabled; keep this service on a trusted/private network.")
        print("Keep Chromium CDP bound to localhost; do not expose port 9222.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    import argparse
    from .cdp import CDPBackend
    from .session import BrowserSession

    parser = argparse.ArgumentParser(description="AGI network Browser Gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cdp-host", default="127.0.0.1")
    parser.add_argument("--cdp-port", type=int, default=9222)
    args = parser.parse_args()

    backend = CDPBackend(args.cdp_host, args.cdp_port)
    environment = BrowserEnvironment(BrowserSession(backend))
    BrowserGateway(environment).serve(args.host, args.port)
