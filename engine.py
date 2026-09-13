#!/usr/bin/env python3
"""Evolution engine: generate, validate, benchmark, activate, repeat."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evaluator import EvaluationError, Evaluator
from model import ModelError, OllamaModel


class EvolutionEngine:
    def __init__(self, root=None):
        self.root = Path(root or __file__).resolve().parent
        self.target = self.root / "program.py"
        self.config = self._load_config()
        self.evaluator = Evaluator(
            self.config.get("test_timeout", 10),
            self.config.get("benchmark_runs", 3),
        )
        self.model = OllamaModel(
            self.config.get("model", "qwen3:4b"),
            self.config.get("ollama_url", "http://127.0.0.1:11434/api/generate"),
            self.config.get("model_timeout", 120),
        )
        self.generation = 0

    def _load_config(self):
        with open(self.root / "config.json", encoding="utf-8") as fh:
            return json.load(fh)

    def _candidate_path(self):
        fd, name = tempfile.mkstemp(prefix="candidate_", suffix=".py", dir=self.root)
        os.close(fd)
        return Path(name)

    def _validate_policy(self, source):
        forbidden = (
            "os.system(",
            "subprocess.",
            "socket.",
            "urllib.",
            "shutil.rmtree(",
        )
        found = [item for item in forbidden if item in source]
        if found:
            raise EvaluationError("candidate uses forbidden capability: " + ", ".join(found))

    def propose(self, instruction):
        source = self.target.read_text(encoding="utf-8")
        candidate_source = self.model.generate(source, instruction)
        self._validate_policy(candidate_source)
        candidate = self._candidate_path()
        candidate.write_text(candidate_source + "\n", encoding="utf-8")
        return candidate

    def activate(self, candidate):
        # Keep the current file only during the atomic transition and launch check.
        old = self.target.with_suffix(".previous.tmp")
        if old.exists():
            old.unlink()
        os.replace(self.target, old)
        try:
            os.replace(candidate, self.target)
            proc = subprocess.run(
                [sys.executable, str(self.target), "--self-test"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.config.get("activation_timeout", 10),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if proc.returncode != 0:
                raise EvaluationError(proc.stderr.strip() or "activated program failed")
        except Exception:
            if self.target.exists():
                self.target.unlink()
            os.replace(old, self.target)
            raise
        old.unlink()

    def iterate(self, instruction):
        self.generation += 1
        candidate = None
        try:
            candidate = self.propose(instruction)
            result = self.evaluator.evaluate(candidate, self.target)
            if not result["accepted"]:
                candidate.unlink(missing_ok=True)
                return {"generation": self.generation, **result}
            self.activate(candidate)
            return {"generation": self.generation, **result, "activated": True}
        except (ModelError, EvaluationError, OSError, ValueError) as exc:
            if candidate is not None:
                candidate.unlink(missing_ok=True)
            return {
                "generation": self.generation,
                "accepted": False,
                "activated": False,
                "reason": str(exc),
            }

    def run_forever(self, instruction):
        while True:
            result = self.iterate(instruction)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            time.sleep(float(self.config.get("iteration_delay", 1)))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--goal",
        default="Improve the program's runtime performance without changing its result semantics.",
    )
    args = parser.parse_args()
    engine = EvolutionEngine()
    if args.once:
        print(json.dumps(engine.iterate(args.goal), ensure_ascii=False, indent=2))
    else:
        engine.run_forever(args.goal)


if __name__ == "__main__":
    main()
