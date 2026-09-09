"""Minimal Chrome DevTools Protocol client implemented with Python standard library only."""
import base64
import http.client
import json
import os
import socket
import struct
import time
import urllib.parse

from .session import BrowserBackend
from .observation import BrowserObservation


class CDPBackend(BrowserBackend):
    """Control an existing Chromium/Chrome instance through its local CDP endpoint."""
    def __init__(self, host="127.0.0.1", port=9222, timeout=15, screenshot_path="data/browser.png"):
        self.host, self.port, self.timeout = host, int(port), int(timeout)
        self.screenshot_path = screenshot_path
        self.ws = None
        self.message_id = 0
        self.url = self.title = self.html = self.text = self.accessibility = ""
        self._connect()

    def _connect(self):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        connection.request("GET", "/json")
        response = connection.getresponse()
        targets = json.loads(response.read().decode("utf-8"))
        connection.close()
        target = next((item for item in targets if item.get("type") == "page" and item.get("webSocketDebuggerUrl")), None)
        if target is None:
            raise RuntimeError("no CDP page target found; start Chromium with --remote-debugging-port")
        self._websocket_connect(target["webSocketDebuggerUrl"])
        self._command("Page.enable")
        self._command("Runtime.enable")

    def _websocket_connect(self, url):
        parsed = urllib.parse.urlparse(url)
        sock = socket.create_connection((parsed.hostname, parsed.port), self.timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        request = (f"GET {path} HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}\r\n"
                   f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n")
        sock.sendall(request.encode("ascii"))
        header = b""
        while b"\r\n\r\n" not in header:
            header += sock.recv(4096)
        if b" 101 " not in header.split(b"\r\n", 1)[0]:
            sock.close()
            raise RuntimeError("CDP websocket handshake failed")
        sock.settimeout(self.timeout)
        self.ws = sock

    def _send_frame(self, payload):
        data = payload.encode("utf-8")
        length = len(data)
        if length < 126:
            header = struct.pack("!BB", 0x81, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", 0x81, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", 0x81, 0x80 | 127, length)
        mask = os.urandom(4)
        self.ws.sendall(header + mask + bytes(v ^ mask[i % 4] for i, v in enumerate(data)))

    def _recv_exact(self, size):
        data = b""
        while len(data) < size:
            chunk = self.ws.recv(size - len(data))
            if not chunk:
                raise RuntimeError("CDP websocket closed")
            data += chunk
        return data

    def _recv_frame(self):
        first, second = self._recv_exact(2)
        opcode, length = first & 0x0F, second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if second & 0x80 else b""
        payload = self._recv_exact(length)
        if mask:
            payload = bytes(v ^ mask[i % 4] for i, v in enumerate(payload))
        if opcode == 8:
            raise RuntimeError("CDP websocket closed")
        if opcode == 9:
            self._send_frame(payload.decode("utf-8"))
            return self._recv_frame()
        return payload.decode("utf-8")

    def _command(self, method, params=None):
        self.message_id += 1
        current = self.message_id
        self._send_frame(json.dumps({"id": current, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self._recv_frame())
            if message.get("id") != current:
                continue
            if "error" in message:
                raise RuntimeError(message["error"].get("message", "CDP error"))
            return message.get("result", {})

    def _evaluate(self, expression):
        result = self._command("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        return result.get("result", {}).get("result", {}).get("value")

    def observe(self):
        self.url = self._evaluate("location.href") or self.url
        self.title = self._evaluate("document.title") or ""
        self.text = self._evaluate("document.body ? document.body.innerText : ''") or ""
        self.html = self._evaluate("document.documentElement ? document.documentElement.outerHTML : ''") or ""
        self.accessibility = self._evaluate("document.body ? document.body.innerText : ''") or ""
        return BrowserObservation(self.url, self.title, self.text, self.html, self.accessibility)

    @staticmethod
    def _find_script(target, selectors):
        q = json.dumps(str(target))
        return f"""(() => {{ const q={q}; const all=[...document.querySelectorAll('{selectors}')];
        const e=document.querySelector(q)||all.find(x=>(x.innerText||x.value||x.getAttribute('aria-label')||x.name||x.placeholder||'').trim()===q);
        if(!e)return false; e.scrollIntoView({{block:'center'}}); return e; }})()"""

    def execute(self, action):
        name, args = action.name, action.args
        if name == "open":
            self._command("Page.navigate", {"url": args["url"]})
            time.sleep(0.5)
        elif name == "click":
            target = json.dumps(str(args["target"]))
            script = f"""(() => {{ const q={target}; const all=[...document.querySelectorAll('button,a,[role],label')]; const e=document.querySelector(q)||all.find(x=>(x.innerText||x.getAttribute('aria-label')||x.name||'').trim()===q); if(!e)return false; e.scrollIntoView({{block:'center'}}); e.click(); return true; }})()"""
            if not self._evaluate(script):
                raise ValueError("browser target not found: " + str(args["target"]))
        elif name == "type":
            target, text = json.dumps(str(args["target"])), json.dumps(str(args["text"]))
            script = f"""(() => {{ const q={target}, v={text}; const all=[...document.querySelectorAll('input,textarea,[contenteditable=true]')]; const e=document.querySelector(q)||all.find(x=>(x.name||x.placeholder||x.getAttribute('aria-label')||'')===q); if(!e)return false; e.focus(); e.value=v; e.dispatchEvent(new Event('input',{{bubbles:true}})); e.dispatchEvent(new Event('change',{{bubbles:true}})); return true; }})()"""
            if not self._evaluate(script):
                raise ValueError("input target not found: " + str(args["target"]))
        elif name == "scroll":
            self._evaluate(f"window.scrollBy(0, {int(args.get('amount', 600))})")
        elif name == "back":
            history = self._command("Page.getNavigationHistory")
            entries, index = history.get("entries", []), history.get("currentIndex", 0)
            if index > 0:
                self._command("Page.navigateToHistoryEntry", {"entryId": entries[index - 1]["id"]})
        elif name == "wait":
            time.sleep(max(0.0, min(float(args.get("seconds", 1)), 30.0)))
        elif name == "extract":
            return self._evaluate("document.body ? document.body.innerText : ''") or ""
        else:
            raise ValueError("unsupported browser action: " + name)
        return self.observe()
