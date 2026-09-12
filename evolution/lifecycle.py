"""Two-process lifecycle for safe self-evolution.

The primary process stays responsible for serving the user. When the user explicitly
requests self-improvement, it starts the independent upgrader and exits. The upgrader
modifies and validates the primary project, starts a fresh primary process only after
validation succeeds, and then exits itself.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path


class EvolutionLifecycle:
    """Coordinate the primary process and the independent upgrader process."""

    def __init__(self, repo_path="/opt/AGI", primary_script="run_browser_gateway.py"):
        self.repo = Path(repo_path).resolve()
        self.primary_script = str(primary_script)
        self._upgrade_requested = False

    @property
    def upgrade_requested(self):
        return self._upgrade_requested

    def request_upgrade(self, prompt):
        """Start the upgrader and schedule shutdown of the current primary process."""
        if self._upgrade_requested:
            return {"accepted": False, "reason": "upgrade already in progress"}

        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("upgrade prompt is required")

        upgrader = self.repo / "run_upgrader.py"
        if not upgrader.is_file():
            raise RuntimeError(f"upgrader entry point not found: {upgrader}")

        self._upgrade_requested = True
        command = [sys.executable, str(upgrader), "--repo", str(self.repo), "--primary", self.primary_script,
                   "--prompt", prompt]
        subprocess.Popen(
            command,
            cwd=self.repo,
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            start_new_session=True,
            close_fds=True,
        )
        return {
            "accepted": True,
            "mode": "two-process-evolution",
            "message": "Upgrade process started. The primary process will shut down now.",
        }

    def schedule_shutdown(self, server, delay=0.15):
        """Stop the HTTP server from a background thread after the current response."""
        def stop():
            import time
            time.sleep(float(delay))
            server.shutdown()

        threading.Thread(target=stop, daemon=True).start()


def start_primary(repo_path, primary_script, arguments=None):
    """Start a new primary process and return immediately."""
    repo = Path(repo_path).resolve()
    script = repo / primary_script
    if not script.is_file():
        raise RuntimeError(f"primary entry point not found: {script}")
    args = list(arguments or [])
    subprocess.Popen(
        [sys.executable, str(script), *args],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        start_new_session=True,
        close_fds=True,
    )
