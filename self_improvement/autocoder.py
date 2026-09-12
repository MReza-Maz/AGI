"""Guarded autonomous code-improvement engine using a local HTTP model.

The model proposes repository changes. This module applies them only after
syntax checks and the project's unit tests pass. Evolution lifecycle code is
protected because it must remain independent from the code it upgrades.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


class AutonomousCoder:
    """Propose, validate, repair, commit, and optionally push an improvement."""

    PROTECTED_PREFIXES = (
        ".git/",
        ".github/workflows/",
        "security/",
        "self_improvement/",
        "evolution/",
        "run_upgrader.py",
    )
    MAX_FILES = 8
    MAX_FILE_BYTES = 200_000
    MAX_CONTEXT_BYTES = 900_000
    MAX_REPAIR_ATTEMPTS = 3

    def __init__(self, repo_path="/opt/AGI", model_url=None, model=None, timeout=180, auto_push=None):
        self.repo = Path(repo_path).resolve()
        self.model_url = model_url or os.environ.get(
            "AGI_MODEL_URL", "http://127.0.0.1:11434/api/generate"
        )
        self.model = model or os.environ.get("AGI_MODEL", "qwen3:4b")
        self.timeout = int(timeout)
        if auto_push is None:
            auto_push = os.environ.get("AGI_AUTO_PUSH", "0").lower() in {
                "1", "true", "yes", "on"
            }
        self.auto_push = bool(auto_push)

    def _run(self, *args, timeout=None):
        return subprocess.run(
            list(args),
            cwd=self.repo,
            text=True,
            capture_output=True,
            timeout=timeout or self.timeout,
        )

    def _files(self):
        result = self._run("git", "ls-files")
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "git ls-files failed")
        return [line for line in result.stdout.splitlines() if line]

    def _snapshot(self):
        files = {}
        total = 0
        for relative in self._files():
            if not relative.endswith((".py", ".json", ".md")) or self._protected(relative):
                continue
            path = self.repo / relative
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")[:30_000]
            except UnicodeDecodeError:
                continue
            size = len(content.encode("utf-8"))
            if total + size > self.MAX_CONTEXT_BYTES:
                break
            files[relative] = content
            total += size
        return files

    @classmethod
    def _protected(cls, path):
        normalized = path.replace("\\", "/").lstrip("./")
        return normalized.startswith(cls.PROTECTED_PREFIXES)

    def _ask_model(self, instruction):
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": instruction,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.model_url,
            data=payload,
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

    def _initial_proposal(self, prompt, files):
        context = "\n\n".join(
            f"FILE: {name}\n{content}" for name, content in files.items()
        )
        instruction = f"""You are the independent coding engineer inside the AGI project.
The primary runtime has stopped and handed control to you.
Improve the primary runtime in response to this explicit user request:
{prompt}

Return ONLY valid JSON with this exact shape:
{{"summary":"...","files":[{{"path":"relative/path.py","content":"complete new file content"}}]}}

Rules:
- Inspect the supplied repository snapshot before changing anything.
- Modify only primary-project code, tests, documentation, and configuration needed for the request.
- Never modify self_improvement/, evolution/, run_upgrader.py, security/, .github/workflows/, .git/.
- Return complete file contents, never diffs or markdown fences.
- Prefer small, coherent changes and preserve compatible public APIs.
- Use Python standard library only unless the repository already uses a dependency.
- Do not add secrets, credentials, malware, persistence, or destructive system commands.
- Add or update tests when appropriate.
- Do not overwrite runtime artifacts or unrelated files.

Repository snapshot:
{context}
"""
        return self._ask_model(instruction)

    def _repair_proposal(self, prompt, proposal, changed_files, validation_error):
        context = "\n\n".join(
            f"FILE: {name}\n{content}" for name, content in changed_files.items()
        )
        instruction = f"""You are repairing an AGI code improvement that failed validation.
The original user request was:
{prompt}

Previous proposal summary:
{proposal.get('summary', '')}

Validation failure:
{validation_error[-20000:]}

Return ONLY valid JSON with this exact shape:
{{"summary":"...","files":[{{"path":"relative/path.py","content":"complete corrected file content"}}]}}

Rules:
- Fix the validation failure while preserving the requested improvement.
- Return complete contents for every file that must be written.
- You may modify only primary-project files. Never modify self_improvement/, evolution/, run_upgrader.py, security/, .github/workflows/, or .git/.
- Use standard-library Python unless the repository already uses a dependency.
- Do not add secrets, credentials, malware, persistence, or destructive commands.

Current candidate files:
{context}
"""
        return self._ask_model(instruction)

    def _backup(self, changed_paths, before_contents):
        """Create a local rollback copy before modifying tracked files."""
        backup_dir = self.repo / "data" / "evolution_backups" / time.strftime(
            "%Y%m%d-%H%M%S"
        )
        backup_dir.mkdir(parents=True, exist_ok=True)
        for relative in changed_paths:
            if relative in before_contents:
                target = backup_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(before_contents[relative], encoding="utf-8")
        return str(backup_dir)

    def _rollback(self, changed_paths, before_contents):
        """Restore files and remove only files created by the proposal."""
        for relative in changed_paths:
            path = self.repo / relative
            if relative in before_contents:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(before_contents[relative], encoding="utf-8")
            elif path.exists():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

    def _validate(self):
        syntax = self._run("python3", "-m", "compileall", "-q", ".")
        tests = self._run(
            "python3", "-m", "unittest", "discover", "-s", "tests", "-v", timeout=max(self.timeout, 300)
        )
        output = "\n".join(
            part for part in (
                "SYNTAX:\n" + (syntax.stderr or syntax.stdout),
                "TESTS:\n" + (tests.stderr or tests.stdout),
            ) if part.strip()
        )
        return syntax.returncode == 0 and tests.returncode == 0, output

    def _push(self):
        result = self._run("git", "push", "origin", "main", timeout=180)
        if result.returncode:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or "git push failed"
            )
        return True

    def _validate_paths(self, changes, untracked):
        if not isinstance(changes, list) or not changes:
            raise ValueError("model proposed no file changes")
        if len(changes) > self.MAX_FILES:
            raise ValueError(f"proposal exceeds {self.MAX_FILES} files")
        normalized = []
        seen = set()
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
        if not self.repo.is_dir() or not (self.repo / ".git").exists():
            raise RuntimeError(f"AGI repository not found: {self.repo}")

        status = self._run("git", "status", "--porcelain")
        if status.returncode:
            raise RuntimeError(status.stderr.strip() or "git status failed")

        tracked_dirty = []
        untracked = set()
        for line in status.stdout.splitlines():
            if not line:
                continue
            code = line[:2]
            path = line[3:]
            if code == "??":
                untracked.add(path.replace("\\", "/"))
            else:
                tracked_dirty.append(path)
        if tracked_dirty:
            raise RuntimeError(
                "repository has uncommitted tracked changes; clean the worktree before self-improvement"
            )

        before_files = set(self._files())
        before_contents = {}
        for relative in before_files:
            path = self.repo / relative
            if path.is_file() and not self._protected(relative):
                try:
                    before_contents[relative] = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass

        proposal = self._initial_proposal(prompt, self._snapshot())
        changed_paths = []
        backup_dir = None
        committed_revision = None

        try:
            for attempt in range(self.MAX_REPAIR_ATTEMPTS + 1):
                changes = proposal.get("files", [])
                normalized = self._validate_paths(changes, untracked)
                changed_paths = [path for path, _ in normalized]
                backup_dir = self._backup(changed_paths, before_contents)

                for relative, content in normalized:
                    path = self.repo / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")

                valid, validation_output = self._validate()
                if valid:
                    break

                self._rollback(changed_paths, before_contents)
                if attempt >= self.MAX_REPAIR_ATTEMPTS:
                    return {
                        "ok": False,
                        "stage": "validation",
                        "error": "validation failed after repair attempts; changes rolled back",
                        "attempts": attempt + 1,
                        "validation": validation_output[-20000:],
                        "changed_paths": changed_paths,
                        "backup": backup_dir,
                    }

                current = {
                    relative: content
                    for relative, content in normalized
                }
                proposal = self._repair_proposal(
                    prompt, proposal, current, validation_output
                )

            message = "autonomous improvement: " + str(
                proposal.get("summary", "update AGI")
            )[:80]
            add = self._run("git", "add", "--", *changed_paths)
            if add.returncode:
                raise RuntimeError(add.stderr.strip() or "git add failed")
            commit = self._run("git", "commit", "-m", message, timeout=180)
            if commit.returncode:
                raise RuntimeError(commit.stderr.strip() or "git commit failed")
            revision = self._run("git", "rev-parse", "HEAD")
            committed_revision = revision.stdout.strip()
            result = {
                "ok": True,
                "stage": "committed",
                "summary": proposal.get("summary", ""),
                "changed_paths": changed_paths,
                "commit": committed_revision,
                "tests": "passed",
                "repair_attempts": attempt,
                "backup": backup_dir,
                "pushed": False,
            }
            if self.auto_push:
                self._push()
                result["pushed"] = True
                result["stage"] = "pushed"
            return result
        except Exception:
            if committed_revision:
                raise
            self._rollback(changed_paths, before_contents)
            self._run("git", "reset", "--", *changed_paths)
            raise
