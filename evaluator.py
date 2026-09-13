#!/usr/bin/env python3
"""Candidate validation and performance evaluation."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path


class EvaluationError(RuntimeError):
    pass


class Evaluator:
    def __init__(self, timeout=10.0, benchmark_runs=3):
        self.timeout = float(timeout)
        self.benchmark_runs = max(1, int(benchmark_runs))

    def syntax_check(self, source: str):
        ast.parse(source)
        return True

    def _run(self, path: Path, timeout=None):
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=timeout or self.timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            raise EvaluationError(proc.stderr.strip() or "candidate exited with non-zero status")
        return elapsed, proc.stdout

    def smoke_test(self, path: Path):
        elapsed, output = self._run(path)
        if not output.strip():
            raise EvaluationError("candidate produced no output")
        return {"elapsed": elapsed, "output": output}

    def benchmark(self, path: Path):
        samples = []
        for _ in range(self.benchmark_runs):
            elapsed, _ = self._run(path)
            samples.append(elapsed)
        return {"samples": samples, "best": min(samples), "average": sum(samples) / len(samples)}

    def evaluate(self, candidate: Path, baseline: Path):
        source = candidate.read_text(encoding="utf-8")
        self.syntax_check(source)
        smoke = self.smoke_test(candidate)
        baseline_smoke = self.smoke_test(baseline)
        if baseline_smoke["output"] != smoke["output"]:
            raise EvaluationError("candidate changed observable output")
        base = self.benchmark(baseline)
        cand = self.benchmark(candidate)
        improved = cand["best"] < base["best"]
        return {
            "accepted": improved,
            "baseline": base,
            "candidate": cand,
            "smoke": smoke,
            "reason": "candidate is faster" if improved else "candidate is not faster",
        }
