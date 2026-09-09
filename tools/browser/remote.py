"""Remote Browser Gateway client using Python standard library only."""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .session import BrowserBackend
from .observation import BrowserObservation


class RemoteBrowserBackend(BrowserBackend):
    """Execute browser actions on a remote Browser Gateway over HTTP JSON."""
    def __init__(self, base_url, token=None, timeout=20):
        self.base_url = str(base_url).rstrip("/")
        self.token = token
        self.timeout = int(timeout)
        self.last_observation = BrowserObservation()

    def _request(self, method, path, payload=None):
        data = None
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read(4 * 1024 * 1024)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"browser gateway HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"browser gateway unavailable: {exc.reason}") from exc
        result = json.loads(body.decode("utf-8"))
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "browser gateway error"))
        return result

    def health(self):
        return self._request("GET", "/health")

    def observe(self):
        result = self._request("GET", "/observe")
        self.last_observation = BrowserObservation(**result["observation"])
        return self.last_observation

    def execute(self, action):
        result = self._request("POST", "/action", action.to_dict())
        value = result.get("result")
        if isinstance(value, dict) and "url" in value:
            self.last_observation = BrowserObservation(**value)
            return self.last_observation
        return value
