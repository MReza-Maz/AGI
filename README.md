# AGI

A pure-Python research architecture for building an AGI-like cognitive system with memory, reasoning, learning, goals, tools, web interaction and controlled self-improvement.

> This project is a research platform, not a claim of artificial general intelligence.

## Design principles

- Python Standard Library only. No pip dependencies.
- The cognitive core uses abstract actions instead of hard-coded website workflows.
- Web access is a first-class environment and can be replaced by other tools later.
- External side effects pass through a permission-aware tool registry.
- Human approval is required before dangerous/self-improvement actions.
- Persistent memory, world state, goals, reasoning and reflection are explicit modules.
- Candidate self-improvements remain isolated from the controller and security layer.

## Architecture

```text
Environment
   -> Perception
   -> Working / Episodic / Semantic Memory
   -> World Model + Goals
   -> Reasoning + Planning + Reflection
   -> Tool Registry
   -> Browser / Terminal / API / Files / VMware / ...
   -> Action
   -> New Observation
   -> Learning + Meta-cognition
   -> Safe Self-improvement
   -> Loop
```

## Web environment

The browser layer is deliberately split from cognition:

```text
Cognitive Core
    |
    +-- browser.open(url)
    +-- browser.click(target)
    +-- browser.type(target, text)
    +-- browser.scroll(amount)
    +-- browser.extract()
    |
Browser Environment
    |
    +-- HTTP backend: read-only web perception
    +-- CDP backend: interactive Chromium/Chrome
```

The CDP backend is implemented with the Python standard library and controls a Chromium/Chrome process exposing its local DevTools Protocol port. Targets can be CSS selectors or visible labels/names where supported.

Start Chromium on a trusted local machine, for example:

```bash
chromium --remote-debugging-port=9222 --user-data-dir=/tmp/agi-browser
```

Then run:

```bash
python3 run_browser.py https://example.com
```

The browser runner observes the page and stores the observation in the agent's memory. Interactive actions are available through the same tool protocol and can later be selected by a planner/model.

## Run the cognitive loop

```bash
python3 run_agent.py --interactive
```

With a goal:

```bash
python3 run_agent.py --goal "Research a technical topic and compare reliable sources" --interactive
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Project layout

```text
AGI/
├── cognition/
│   ├── agent.py
│   ├── goals.py
│   └── world_model.py
├── core/
├── memory/
├── reasoning/
├── learning/
├── self_improvement/
├── security/
├── tools/
│   ├── registry.py
│   └── browser/
│       ├── actions.py
│       ├── observation.py
│       ├── session.py
│       ├── environment.py
│       └── cdp.py
├── tests/
│   └── test_tools.py
├── config.json
├── run_agent.py
└── run_browser.py
```

## Current status

v0.4: the cognitive runtime now has a first-class browser environment, abstract browser actions, structured observations, a permission-aware tool registry, and a standard-library-only Chromium CDP backend.

## Next milestones

1. Connect planning/model output to structured tool actions.
2. Add accessibility-tree extraction and screenshot perception to the CDP backend.
3. Add browser task benchmarks and recovery/evaluation loops.
4. Add a stronger retrieval and memory consolidation system.
5. Add model-based world simulation and hypothesis testing.
6. Add continual learning driven by task outcomes.
7. Add more tools through the same capability protocol: filesystem, terminal, REST, database, SSH, Docker and VMware.
8. Strengthen sandboxing, policy enforcement, audit trails and rollback.
