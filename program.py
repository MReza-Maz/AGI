#!/usr/bin/env python3
"""The mutable program. This file is intentionally designed to evolve."""

VERSION = 1


def workload(size=10000):
    """Return a deterministic workload result used for correctness checks."""
    total = 0
    for value in range(size):
        total += value * value
    return total


def run():
    result = workload()
    print(f"program version={VERSION} result={result}")
    return result


def self_test():
    expected = 333283335000
    return workload() == expected


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        if not self_test():
            raise SystemExit(1)
        print("self-test: ok")
    else:
        run()
