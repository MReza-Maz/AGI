"""Safe direct self-improvement engine.

The engine allows the local reasoning model to propose source-code changes,
applies them directly to the AGI repository, validates the result, and
automatically rolls back invalid changes.

Git is intentionally not part of the runtime improvement mechanism.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


class AutonomousCoder:
    MAX_FILES = 8
    MAX_FILE_BYTES = 200_000
    MAX_CONTEXT_BYTES = 900_000
    MAX_REPAIR_ATTEMPTS = 3

    PROTECTED_PREFIXES = (
        ".git/",
        ".github/workflows/",
        "security/",
        "self_improvement/",
        "evolution/",
        "run_upgrader.py",
    )

    RUNTIME_PREFIXES = (
        "checkpoints/",
        "logs/",
        "data/evolution_backups/",
    )

    RUNTIME_FILES = (
        ".env",
        ".env.local",
        ".env.production",
    )

    def __init__(self, repo_path="/opt/AGI", model="qwen3:4b", ollama_url="http://127.0.0.1:11434/api/generate", timeout=300):
        self.repo = Path(repo_path).resolve()
        self.model = model or os.environ.get("AGI_MODEL", "qwen3:4b")
        self.ollama_url = ollama_url or os.environ.get("AGI_MODEL_URL", "http://127.0.0.1:11434/api/generate")
        self.timeout = int(timeout)

        if not self.repo.is_dir():
            raise RuntimeError(f"Repository does not exist: {self.repo}")

    def _normalize(self, path):
        path = str(path).replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        return path.lstrip("/")

    def _protected(self, path):
        path = self._normalize(path)
        if path in self.RUNTIME_FILES:
            return True
        for prefix in self.PROTECTED_PREFIXES + self.RUNTIME_PREFIXES:
            if path == prefix.rstrip("/") or path.startswith(prefix):
                return True
        return False

    def _safe_path(self, relative):
        relative = self._normalize(relative)
        if not relative or ".." in Path(relative).parts:
            raise ValueError(f"invalid path: {relative}")
        if self._protected(relative):
            raise ValueError(f"protected path cannot be modified: {relative}")
        path = (self.repo / relative).resolve()
        try:
            path.relative_to(self.repo)
        except ValueError as exc:
            raise ValueError(f"path escapes repository: {relative}") from exc
        return path

    def _files(self):
        result = []
        for path in self.repo.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.repo).as_posix()
            if ".git" in Path(relative).parts or "__pycache__" in Path(relative).parts:
                continue
            result.append(path)
        return sorted(result)

    def _snapshot(self):
        snapshot = {}
        total = 0
        allowed = {".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"}
        for path in self._files():
            relative = path.relative_to(self.repo).as_posix()
            if self._protected(relative) or path.suffix.lower() not in allowed:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(content.encode("utf-8")) > self.MAX_FILE_BYTES:
                continue
            size = len(content.encode("utf-8"))
            if total + size > self.MAX_CONTEXT_BYTES:
                break
            snapshot[relative] = content
            total += size
        return snapshot

    def _ask_model(self, prompt):
        payload = {"model": self.model, "prompt": prompt, "stream": False, "format": "json"}
        request = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data.get("response", data.get("message", {}).get("content", ""))
        if not text:
            raise RuntimeError("coding model returned an empty response")
        try:
            proposal = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"coding model returned invalid JSON: {exc}") from exc
        if not isinstance(proposal, dict):
            raise RuntimeError("coding model returned an invalid proposal")
        return proposal

    def _initial_proposal(self, goal, snapshot):
        context = "\n\n".join(f"FILE: {name}\n{content}" for name, content in snapshot.items())
        return self._ask_model(f'''You are the software engineering component of an autonomous AGI.

User improvement request:
{goal}

Analyze the repository and propose the smallest coherent implementation that satisfies the request.

Return ONLY valid JSON:
{{"summary":"short explanation","files":[{{"path":"relative/path.py","content":"complete new file content"}}]}}

Rules:
- Return complete file contents, never diffs or markdown fences.
- Preserve existing working functionality and public APIs where possible.
- Do not modify self_improvement/, evolution/, run_upgrader.py, security/, .github/workflows/, or .git/.
- Do not modify checkpoints, logs, runtime backups, credentials, or secrets.
- Do not add external Python dependencies; prefer the standard library.
- Add or update tests when appropriate.

Repository snapshot:
{context}
''')

    def _repair_proposal(self, goal, proposal, candidate_files, validation_output):
        context = "\n\n".join(f"FILE: {name}\n{content}" for name, content in candidate_files.items())
        return self._ask_model(f'''Repair this autonomous AGI code improvement.

Original request:
{goal}

Previous summary:
{proposal.get("summary", "")}

Validation failure:
{validation_output[-20000:]}

Return ONLY valid JSON:
{{"summary":"repair description","files":[{{"path":"relative/path.py","content":"complete corrected file content"}}]}}

Never modify self_improvement/, evolution/, run_upgrader.py, security/, .github/workflows/, .git/, checkpoints, logs, runtime backups, credentials, or secrets.
Use standard-library Python unless the repository already uses a dependency.

Current candidate files:
{context}
''')

    def _backup(self, changed_paths, before_contents):
        backup_dir = self.repo / "data" / "evolution_backups" / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.time_ns()}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        metadata = []
        for relative in changed_paths:
            if relative in before_contents:
                target = backup_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(before_contents[relative], encoding="utf-8")
                metadata.append({"path": relative, "existed": True})
            else:
                metadata.append({"path": relative, "existed": False})
        (backup_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return str(backup_dir)

    def _rollback(self, changed_paths, before_contents):
        for relative in changed_paths:
            path = self._safe_path(relative)
            if relative in before_contents:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(before_contents[relative], encoding="utf-8")
            elif path.exists():
                path.unlink() if path.is_file() or path.is_symlink() else shutil.rmtree(path)

    def restore_backup(self, backup_dir, changed_paths):
        backup = Path(backup_dir)
        if not backup.is_dir():
            return {"ok": False, "reason": "backup directory not found"}
        restored = []
        for relative in changed_paths:
            source = backup / relative
            target = self._safe_path(relative)
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif target.exists():
                target.unlink() if target.is_file() or target.is_symlink() else shutil.rmtree(target)
            restored.append(relative)
        return {"ok": True, "restored": restored}

    def _validate(self):
        compile_process = subprocess.run(["python3", "-m", "compileall", "-q", "."], cwd=self.repo, capture_output=True, text=True, timeout=180)
        compile_output = compile_process.stdout + compile_process.stderr
        if compile_process.returncode != 0:
            return {"ok": False, "stage": "compile", "output": compile_output}
        test_process = subprocess.run(["python3", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=self.repo, capture_output=True, text=True, timeout=600)
        return {"ok": test_process.returncode == 0, "stage": "tests", "output": test_process.stdout + test_process.stderr}

    def _validate_paths(self, changes, untracked=None):
        """Validate proposed paths while preserving the public test API."""
        if not isinstance(changes, list) or not changes:
            raise ValueError("model proposed no file changes")
        if len(changes) > self.MAX_FILES:
            raise ValueError(f"proposal exceeds {self.MAX_FILES} files")
        untracked = {self._normalize(path) for path in (untracked or set())}
        normalized, seen = [], set()
        for item in changes:
            if not isinstance(item, dict):
                raise ValueError("invalid model file change")
            relative = self._normalize(item.get("path", ""))
            content = item.get("content")
            if not relative or not isinstance(content, str):
                raise ValueError("invalid model file change")
            if relative in seen:
                raise ValueError(f"duplicate file in proposal: {relative}")
            if self._protected(relative) or ".." in Path(relative).parts:
                raise ValueError(f"protected or invalid path: {relative}")
            if len(content.encode("utf-8")) > self.MAX_FILE_BYTES:
                raise ValueError(f"file too large: {relative}")
            if relative in untracked:
                raise ValueError(f"proposal would overwrite untracked file: {relative}")
            self._safe_path(relative)
            seen.add(relative)
            normalized.append((relative, content))
        return normalized

    def improve(self, goal):
        goal = str(goal).strip()
        if not goal:
            raise ValueError("improvement goal is required")

        before_contents = {}
        for path in self._files():
            relative = path.relative_to(self.repo).as_posix()
            if self._protected(relative):
                continue
            try:
                before_contents[relative] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

        proposal = self._initial_proposal(goal, self._snapshot())
        last_error = None
        for attempt in range(self.MAX_REPAIR_ATTEMPTS + 1):
            changed_paths = []
            backup_dir = None
            try:
                normalized = self._validate_paths(proposal.get("files", []), set())
                changed_paths = [relative for relative, _ in normalized]
                backup_dir = self._backup(changed_paths, before_contents)
                for relative, content in normalized:
                    path = self._safe_path(relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(f".{path.name}.evolution.tmp")
                    temporary.write_text(content, encoding="utf-8")
                    os.replace(temporary, path)
                validation = self._validate()
                if validation["ok"]:
                    return {"ok": True, "stage": "applied", "attempt": attempt + 1, "summary": proposal.get("summary", ""), "changed_paths": changed_paths, "backup": backup_dir, "tests": "passed", "git": False}
                last_error = validation
                self._rollback(changed_paths, before_contents)
                if attempt >= self.MAX_REPAIR_ATTEMPTS:
                    return {"ok": False, "stage": "rolled_back", "attempts": attempt + 1, "error": "validation failed after repair attempts", "validation": validation["output"][-20000:], "changed_paths": changed_paths, "backup": backup_dir, "git": False}
                proposal = self._repair_proposal(goal, proposal, {relative: content for relative, content in normalized}, validation["output"])
            except Exception as exc:
                if changed_paths:
                    self._rollback(changed_paths, before_contents)
                raise RuntimeError(f"self-improvement failed: {exc}") from exc

        return {"ok": False, "stage": "rolled_back", "error": last_error, "git": False}
