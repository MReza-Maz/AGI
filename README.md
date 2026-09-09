# AGI

A pure-Python research architecture for building an AGI-like cognitive system with controlled self-improvement.

> This project is a research platform, not a claim of artificial general intelligence.

## Design principles

- Python Standard Library only. No pip dependencies.
- Human approval is required before self-improvement proposals can affect the active system.
- The controller, security layer and audit trail are protected from agent modification.
- Every accepted change should have a checkpoint and rollback path.
- Learning, memory, reasoning and self-improvement are explicit modules.

## Architecture

Perception -> Workspace -> Memory -> Reasoning -> World Model -> Goals -> Planner -> Executor -> Feedback -> Learning

Self-improvement runs separately:

Analyze -> Proposal -> Candidate -> Sandbox -> Tests -> Benchmark -> Human Approval -> Deploy/Reject -> Rollback

## Requirements

Python 3.11+ recommended. The code intentionally uses only the Python standard library.

## Run

```bash
python3 main.py
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Project layout

```text
AGI/
├── main.py
├── config.json
├── core/
│   ├── tensor.py
│   ├── nn.py
│   ├── attention.py
│   ├── transformer.py
│   └── brain.py
├── memory/
│   ├── working.py
│   ├── episodic.py
│   ├── semantic.py
│   └── manager.py
├── reasoning/
│   ├── planner.py
│   ├── reflection.py
│   └── engine.py
├── learning/
│   ├── loss.py
│   ├── optimizer.py
│   ├── dataset.py
│   └── trainer.py
├── self_improvement/
│   ├── analyzer.py
│   ├── candidate.py
│   ├── sandbox.py
│   └── controller.py
├── security/
│   ├── approval.py
│   ├── audit.py
│   └── checkpoint.py
└── tests/
```

## Status

v0.2 foundation: deterministic cognitive loop, persistent memory, small neural components, training primitives, and a controlled self-improvement pipeline.

## Roadmap

1. Better tensor/autograd coverage.
2. Tokenization and sequence datasets.
3. More capable transformer blocks.
4. Retrieval-augmented semantic memory.
5. World-model state and simulation.
6. Goal prioritization and multi-step planning.
7. Evaluator-driven continual learning.
8. Sandboxed candidate execution and stronger policy enforcement.
9. Distributed training and hardware acceleration through optional future backends.
