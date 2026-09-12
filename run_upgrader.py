"""Independent upgrader process for the two-process AGI self-evolution loop."""
import argparse
import json
import sys
import time
from pathlib import Path

from evolution.lifecycle import start_primary
from self_improvement.autocoder import AutonomousCoder


def write_status(repo, payload):
    path = Path(repo) / "data" / "evolution_last.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="AGI independent self-upgrader")
    parser.add_argument("--repo", default="/opt/AGI")
    parser.add_argument("--primary", default="run_browser_gateway.py")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-timeout", type=int, default=300)
    parser.add_argument("--push", action="store_true")
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
        result["finished_at"] = time.time()
        write_status(args.repo, result)

        # The primary was stopped before this process started. Restore service even
        # when the proposed upgrade failed and the repository was rolled back.
        start_primary(args.repo, args.primary)
        return 0 if result.get("ok") else 2
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
