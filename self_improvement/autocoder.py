"""Direct autonomous code-improvement engine using a local HTTP model."""
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


class AutonomousCoder:
    """Propose, validate, repair, and directly apply project improvements."""

    PROTECTED_PREFIXES = (".git/", ".github/workflows/", "security/", "self_improvement/", "evolution/", "run_upgrader.py")
    MAX_FILES = 8
    MAX_FILE_BYTES = 200_000
    MAX_CONTEXT_BYTES = 900_000
    MAX_REPAIR_ATTEMPTS = 3

    def __init__(self, repo_path="/opt/AGI", model_url=None, model=None, timeout=180, auto_push=None):
        self.repo = Path(repo_path).resolve()
        self.model_url = model_url or os.environ.get("AGI_MODEL_URL", "http://127.0.0.1:11434/api/generate")
        self.model = model or os.environ.get("AGI_MODEL", "qwen3:4b")
        self.timeout = int(timeout)

    def _run(self, *args, timeout=None):
        return subprocess.run(list(args), cwd=self.repo, text=True, capture_output=True, timeout=timeout or self.timeout)

    def _files(self):
        files = []
        for path in self.repo.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.repo).as_posix()
            if ".git" in Path(relative).parts or "__pycache__" in Path(relative).parts:
                continue
            files.append(relative)
        return files

    def _snapshot(self):
        files, total = {}, 0
        for relative in self._files():
            if not relative.endswith((".py", ".json", ".md")) or self._protected(relative):
                continue
            try:
                content = (self.repo / relative).read_text(encoding="utf-8")[:30_000]
            except (UnicodeDecodeError, OSError):
                continue
            size = len(content.encode("utf-8"))
            if total + size > self.MAX_CONTEXT_BYTES:
                break
            files[relative] = content
            total += size
        return files

    @classmethod
    def _protected(cls, path):
        normalized = str(path).replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized.lstrip("/")
        for protected in cls.PROTECTED_PREFIXES:
            if protected.endswith("/"):
                if normalized == protected[:-1] or normalized.startswith(protected):
                    return True
            elif normalized == protected:
                return True
        return False

    def _ask_model(self, instruction):
        payload = json.dumps({"model": self.model, "prompt": instruction, "stream": False, "format": "json"}).encode("utf-8")
        request = urllib.request.Request(self.model_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
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

    def _initial_proposal(self, prompt, files):
        context = "\n\n".join(f"FILE: {name}\n{content}" for name, content in files.items())
        return self._ask_model(f'''You are the independent coding engineer inside the AGI project.
Improve the primary runtime in response to this explicit user request:
{prompt}

Return ONLY valid JSON with this exact shape:
{{"summary":"...","files":[{{"path":"relative/path.py","content":"complete new file content"}}]}}

Rules:
- Inspect the repository snapshot before changing anything.
- Modify only primary-project code, tests, documentation, and configuration needed for the request.
- Never modify self_improvement/, evolution/, run_upgrader.py, security/, .github/workflows/, or .git/.
- Return complete file contents, never diffs or markdown fences.
- Prefer small coherent changes and preserve compatible public APIs.
- Use Python standard library unless the repository already uses a dependency.
- Do not add secrets, credentials, malware, persistence, or destructive commands.
- Add or update tests when appropriate.

Repository snapshot:
{context}
''')

    def _repair_proposal(self, prompt, proposal, changed_files, validation_error):
        context = "\n\n".join(f"FILE: {name}\n{content}" for name, content in changed_files.items())
        return self._ask_model(f'''You are repairing an AGI code improvement that failed validation.
Original request:
{prompt}
Previous summary:
{proposal.get("summary", "")}
Validation failure:
{validation_error[-20000:]}

Return ONLY valid JSON with this exact shape:
{{"summary":"...","files":[{{"path":"relative/path.py","content":"complete corrected file content"}}]}}

Never modify self_improvement/, evolution/, run_upgrader.py, security/, .github/workflows/, or .git/.
Use standard-library Python unless the repository already uses a dependency.

Current candidate files:
{context}
''')

    def _backup(self, changed_paths, before_contents):
        backup_dir = self.repo / "data" / "evolution_backups" / time.strftime("%Y%m%d-%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for relative in changed_paths:
            if relative in before_contents:
                target = backup_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(before_contents[relative], encoding="utf-8")
        return str(backup_dir)

    def _rollback(self, changed_paths, before_contents):
        for relative in changed_paths:
            path = self.repo / relative
            if relative in before_contents:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(before_contents[relative], encoding="utf-8")
            elif path.exists():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                else:
                    shutil.rmtree(path)

    def restore_backup(self, backup_dir, changed_paths):
        backup = Path(backup_dir)
        for relative in changed_paths:
            source, target = backup / relative, self.repo / relative
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif target.exists():
                if target.is_file() or target.is_symlink():
                    target.unlink()
                else:
                    shutil.rmtree(target)

    def _validate(self):
        syntax = self._run("python3", "-m", "compileall", "-q", ".")
        tests = self._run("python3", "-m", "unittest", "discover", "-s", "tests", "-v", timeout=max(self.timeout, 300))
        output = "\n".join(("SYNTAX:\n" + (syntax.stderr or syntax.stdout), "TESTS:\n" + (tests.stderr or tests.stdout)))
        return syntax.returncode == 0 and tests.returncode == 0, output

    def _validate_paths(self, changes, untracked=None):
        if not isinstance(changes, list) or not changes:
            raise ValueError("model proposed no file changes")
        if len(changes) > self.MAX_FILES:
            raise ValueError(f"proposal exceeds {self.MAX_FILES} files")
        untracked = set(untracked or ())
        normalized, seen = [], set()
        for item in changes:
            if not isinstance(item, dict):
                raise ValueError("invalid model file change")
            relative = str(item.get("path", "")).replace("\\", "/").lstrip("/")
            content = item.get("content")
            if not relative or not isinstance(content, str):
                raise ValueError("invalid model file change")
            if self._protected(relative) or ".." in Path(relative).parts:
                raise ValueError(f"protected or invalid path: {relative}")
            if relative in seen:
                raise ValueError(f"duplicate file in proposal: {relative}")
            if len(content.encode("utf-8")) > self.MAX_FILE_BYTES:
                raise ValueError(f"file too large: {relative}")
            if relative in untracked:
                raise ValueError(f"proposal would overwrite untracked file: {relative}")
            seen.add(relative)
            normalized.append((relative, content))
        return normalized

    def improve(self, prompt):
        if not self.repo.is_dir():
            raise RuntimeError(f"AGI repository not found: {self.repo}")
        before_contents = {}
        for relative in self._files():
            if self._protected(relative):
                continue
            path = self.repo / relative
            try:
                before_contents[relative] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                pass
        proposal = self._initial_proposal(prompt, self._snapshot())
        changed_paths, backup_dir = [], None
        for attempt in range(self.MAX_REPAIR_ATTEMPTS + 1):
            try:
                normalized = self._validate_paths(proposal.get("files", []))
                changed_paths = [path for path, _ in normalized]
                backup_dir = self._backup(changed_paths, before_contents)
                for relative, content in normalized:
                    path = self.repo / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                valid, validation_output = self._validate()
                if valid:
                    return {"ok": True, "stage": "applied", "summary": proposal.get("summary", ""), "changed_paths": changed_paths, "tests": "passed", "repair_attempts": attempt, "backup": backup_dir, "git": False}
                self._rollback(changed_paths, before_contents)
                if attempt >= self.MAX_REPAIR_ATTEMPTS:
                    return {"ok": False, "stage": "validation", "error": "validation failed after repair attempts; changes rolled back", "attempts": attempt + 1, "validation": validation_output[-20000:], "changed_paths": changed_paths, "backup": backup_dir, "git": False}
                proposal = self._repair_proposal(prompt, proposal, {relative: content for relative, content in normalized}, validation_output)
            except Exception:
                self._rollback(changed_paths, before_contents)
                raise
