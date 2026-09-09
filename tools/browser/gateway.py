"""Network gateway and web UI for the AGI browser environment.

Authentication is intentionally omitted because this gateway is designed for a
trusted/private network. Do not expose it directly to an untrusted network.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .actions import BrowserAction
from .agent import BrowserPromptAgent
from .environment import BrowserEnvironment
from .web_ui import HTML
from self_improvement.autocoder import AutonomousCoder


class BrowserGateway:
    """HTTP gateway around a BrowserEnvironment with a prompt-driven web UI."""
    def __init__(self, environment, max_body=64 * 1024, repo_path="/opt/AGI"):
        self.environment = environment
        self.max_body = int(max_body)
        self.prompt_agent = BrowserPromptAgent(environment)
        self.autocoder = AutonomousCoder(repo_path=repo_path)

    @staticmethod
    def _response_text(observation):
        if not isinstance(observation, dict):
            return str(observation)
        title = observation.get("title") or "Untitled page"
        url = observation.get("url") or ""
        text = str(observation.get("text") or "").strip()
        if len(text) > 4000:
            text = text[:4000] + "..."
        return f"Page: {title}\nURL: {url}\n\n{text}" if text else f"Page: {title}\nURL: {url}"

    @staticmethod
    def _is_self_improvement(prompt):
        text = prompt.lower()
        phrases = (
            "upgrade yourself", "improve yourself", "self improve", "self-improvement",
            "upgrade your code", "improve your code", "ارتقا", "خودت را ارتقا", "خودشو ارتقا",
            "کد خودت", "کد خودتو", "خودت را بهتر", "خودشو بهتر",
        )
        return any(phrase in text for phrase in phrases)

    def handler_class(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AGI-Browser-Gateway/2.1"

            def _json(self, status, payload):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _html(self):
                data = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
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
                    if path in ("", "/"):
                        return self._html()
                    if path == "/health":
                        return self._json(200, {"ok": True, "service": "browser-gateway", "version": "2.1", "auth": False, "ui": True, "self_improvement": True})
                    if path in ("/observe", "/api/observe"):
                        return self._json(200, {"ok": True, "observation": gateway.environment.observe()})
                    if path == "/tools":
                        return self._json(200, {"ok": True, "actions": ["open", "click", "type", "scroll", "back", "wait", "extract", "self-improve"]})
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
                    if path == "/api/prompt":
                        prompt = payload.get("prompt")
                        if not isinstance(prompt, str) or not prompt.strip():
                            raise ValueError("prompt is required")
                        prompt = prompt.strip()
                        if gateway._is_self_improvement(prompt):
                            result = gateway.autocoder.improve(prompt)
                            return self._json(200, {
                                "ok": result.get("ok", False),
                                "mode": "self-improvement",
                                "response": result.get("summary", result.get("error", "self-improvement failed")),
                                "observation": gateway.environment.observe(),
                                "steps": [{"type": "self-improvement", "result": result}],
                            })
                        result = gateway.prompt_agent.run(prompt)
                        observation = result.get("observation", {})
                        return self._json(200, {"ok": True, "mode": "browser", "response": gateway._response_text(observation), "steps": result.get("steps", []), "observation": observation})
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
