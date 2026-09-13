# Self-Evolving Program

A small Python research project that lets a program generate, test, benchmark, and activate improved versions of itself.

## Cycle

1. Read `program.py`.
2. Ask the local Ollama model for a complete replacement.
3. Reject forbidden capabilities.
4. Run syntax and smoke checks.
5. Benchmark candidate against the current program.
6. Activate only when the candidate is faster and preserves observable output.
7. Launch the activated version with `--self-test`.
8. Roll back if activation fails.
9. Repeat.

The evolution engine is separate from the mutable target program. Git is not part of the runtime evolution loop.
