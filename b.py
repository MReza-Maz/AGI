#!/usr/bin/env python3
"""Self-evolving worker B.

This file and a.py form a two-process evolution chain. Each worker improves
its partner, validates the replacement, runs it, and then exits so the new
worker becomes the only active process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MODEL = "qwen3:4b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_TIMEOUT = 600
MODEL_TOKENS = 4096
BENCHMARK_RUNS = 5


def workload(size: int = 50000) -> int:
    total = 0
    for value in range(size):
        total += value * value
    return total


def self_test() -> None:
    expected = sum(value * value for value in range(1000))
    if workload(1000) != expected:
        raise RuntimeError("workload self-test failed")


def call_model(source: str, target_name: str) -> str:
    prompt = f'''You are the evolution engine for a two-file self-improving Python system.

The target file is {target_name}. Produce a COMPLETE replacement for that file.

CURRENT TARGET SOURCE:
```python
{source}
```

Your replacement must preserve this protocol:
1. When started normally, this file reads and improves its sibling Python file.
2. It validates the generated replacement before activation.
3. It runs the replacement after successful activation.
4. It exits after handing control to the replacement.
5. The two files must therefore continue alternating forever until the user stops the chain.
6. --self-test must perform internal correctness checks and exit with code 0 when correct.
7. --workload must perform the deterministic benchmark workload and print ONLY its integer result.

Improve the implementation as aggressively as useful: algorithmic efficiency, execution speed,
code quality, robustness, reasoning about the evolution process, and reliability may all be improved.
You may rewrite the complete file. Do not explain anything.
Return ONLY valid standalone Python source code.
'''
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": MODEL_TOKENS},
            "keep_alive": "10m",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=MODEL_TIMEOUT) as response:
        data = json.loads(response.read().decode("utf-8"))
    result = str(data.get("response", "")).strip()
    if not result:
        raise RuntimeError("model returned empty source")
    if "```" in result:
        parts = result.split("```")
        if len(parts) >= 3:
            result = parts[1].strip()
            if result.startswith("python"):
                result = result[6:].lstrip()
    return result.strip() + "\n"


def run_script(path: Path, *args: str, timeout: float = 30.0) -> tuple[float, str]:
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, str(path), *args],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or f"{path.name} exited with code {process.returncode}")
    return elapsed, process.stdout


def benchmark(path: Path) -> float:
    samples = []
    for _ in range(BENCHMARK_RUNS):
        elapsed, output = run_script(path, "--workload")
        if not output.strip().lstrip("-").isdigit():
            raise RuntimeError(f"invalid workload output from {path.name}: {output!r}")
        samples.append(elapsed)
    return min(samples)


def candidate_path(root: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix="evolution_", suffix=".py", dir=root)
    os.close(fd)
    return Path(name)


def improve_partner() -> None:
    current = Path(__file__).resolve()
    root = current.parent
    partner = root / ("b.py" if current.name == "a.py" else "a.py")
    candidate = candidate_path(root)

    try:
        source = partner.read_text(encoding="utf-8")
        generated = call_model(source, partner.name)
        candidate.write_text(generated, encoding="utf-8")

        compile(generated, str(candidate), "exec")

        _, candidate_test = run_script(candidate, "--self-test")
        _, baseline_test = run_script(partner, "--self-test")
        if not candidate_test.strip() or not baseline_test.strip():
            raise RuntimeError("self-test produced no output")

        _, candidate_output = run_script(candidate, "--workload")
        _, baseline_output = run_script(partner, "--workload")
        if candidate_output != baseline_output:
            raise RuntimeError("candidate changed workload result")

        baseline_time = benchmark(partner)
        candidate_time = benchmark(candidate)
        print(
            json.dumps(
                {
                    "worker": current.name,
                    "target": partner.name,
                    "baseline_seconds": baseline_time,
                    "candidate_seconds": candidate_time,
                    "accepted": candidate_time < baseline_time,
                },
                indent=2,
            ),
            flush=True,
        )

        if candidate_time >= baseline_time:
            return

        backup = partner.with_suffix(".old")
        if backup.exists():
            backup.unlink()
        os.replace(partner, backup)
        try:
            os.replace(candidate, partner)
            run_script(partner, "--self-test")
        except Exception:
            if partner.exists():
                partner.unlink()
            os.replace(backup, partner)
            raise
        backup.unlink()

        subprocess.Popen(
            [sys.executable, str(partner)],
            cwd=root,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        raise SystemExit(0)
    finally:
        if candidate.exists():
            candidate.unlink()


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        print("self-test: ok")
        return
    if "--workload" in sys.argv:
        print(workload())
        return
    improve_partner()


if __name__ == "__main__":
    main()
