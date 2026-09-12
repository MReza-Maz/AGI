"""Independent upgrader process for the two-process AGI self-evolution loop."""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from evolution.lifecycle import start_primary
from self_improvement.autocoder import AutonomousCoder


def write_status(repo, payload):
    path = Path(repo) / "data" / "evolution_last.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_for_process_exit(pid, timeout=30):
    """Wait until the primary process is gone before touching its project files."""
    if not pid:
        return
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(0.1)
    raise RuntimeError(f"primary process {pid} did not stop within {timeout} seconds")


def wait_for_health(url, timeout=30):
    """Wait for the restarted primary HTTP endpoint to become healthy."""
    deadline = time.time() + float(timeout)
    last_error = "health check not completed"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and payload.get("ok") is True:
                    return payload
                last_error = f"unexpected health response: {payload}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"primary health check failed: {last_error}")


def git_revert(repo, revision):
    """Revert a committed upgrade without rewriting repository history."""
    result = subprocess.run(
        ["git", "revert", "--no-edit", revision],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git revert failed")
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="AGI independent self-upgrader")
    parser.add_argument("--repo", default="/opt/AGI")
    parser.add_argument("--primary", default="run_browser_gateway.py")
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-timeout", type=int, default=300)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--health-url", default="http://127.0.0.1:8080/health")
    parser.add_argument("--health-timeout", type=int, default=30)
    args = parser.parse_args()

    result = {
        "started_at": time.time(),
        "prompt": args.prompt,
        "ok": False,
        "stage": "starting",
        "primary": args.primary,
    }
    write_status(args.repo, result)

    try:
        wait_for_process_exit(args.wait_pid)
        result["stage"] = "primary-stopped"
        write_status(args.repo, result)

        coder = AutonomousCoder(
            repo_path=args.repo,
            model_url=args.model_url,
            model=args.model,
            timeout=args.model_timeout,
            auto_push=args.push,
        )
        result = coder.improve(args.prompt)
        result["prompt"] = args.prompt
        result["primary"] = args.primary
        write_status(args.repo, result)

        if not result.get("ok"):
            start_primary(args.repo, args.primary)
            return 2

        revision = result.get("commit")
        result["stage"] = "restarting"
        write_status(args.repo, result)
        start_primary(args.repo, args.primary)

        try:
            health = wait_for_health(args.health_url, args.health_timeout)
            result["health"] = health
            result["stage"] = "verified"
            result["finished_at"] = time.time()
            write_status(args.repo, result)
            return 0
        except Exception as health_error:
            result["stage"] = "rollback"
            result["health_error"] = str(health_error)
            write_status(args.repo, result)
            if not revision:
                raise
            git_revert(args.repo, revision)
            result["rollback"] = "reverted"
            result["stage"] = "rollback-complete"
            write_status(args.repo, result)
            start_primary(args.repo, args.primary)
            return 3

    except Exception as exc:
        result = {
            "ok": False,
            "stage": "upgrader-error",
            "error": str(exc),
            "prompt": args.prompt,
            "primary": args.primary,
            "finished_at": time.time(),
        }
        write_status(args.repo, result)
        try:
            start_primary(args.repo, args.primary)
        except Exception as restart_error:
            result["restart_error"] = str(restart_error)
            write_status(args.repo, result)
        return 1


if __name__ == "__main__":
    sys.exit(main())
