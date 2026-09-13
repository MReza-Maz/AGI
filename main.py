#!/usr/bin/env python3
"""Entry point for the autonomous self-improvement engine."""

from engine import EvolutionEngine


if __name__ == "__main__":
    EvolutionEngine().run_forever(
        "Improve the target program's runtime performance while preserving its behavior."
    )
