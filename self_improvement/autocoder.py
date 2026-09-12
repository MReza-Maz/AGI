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
    """Read the repository, propose changes, validate them, commit, and optionally push."""

    PROTECTED_PREFIXES = (
        ".git/",
        ".github/workflows/",
        "security/",
        "self_improvement/autocoder.py",
        "evolution/",
        "run_upgrader.py",
    )
    MAX_FILES = 8
    MAX_FILE_BYTES = 200_000

    def __init__(self, repo_path="/opt/AGI", model_url=None, model=None, timeout=180, auto_push=None):
        self.repo = Path(repo_path).resolve()
        self.model_url = model_url or os.environ.get("AGI_MODEL_URL", "http://127.0.0.1:11434/api/generate")
        self.model = model or os.environ.get("AGI_MODEL", "qwen3:4b")
        self.timeout = int(timeout)
        if auto_push is None:
            auto_push = os.environ.get("AGI_AUTO_PUSH", "0").lower() in {"1", "true", "yes", "on"}
        self.auto_push = bool(auto_push)

    def _run(self, *args):
        return subprocess.run(
            list(args), cwd=self.repo, text=True, capture_output=True, timeout=self.timeout
        )

    def _files(self):
        result = self._run("git", "ls-files")
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "git ls-files failed")
        return [line for line in result.stdout.splitlines() if line]

    def _snapshot(self):
        files = {}
        for relative in self._files():
            if relative.endswith((".py", ".json", ".md")) and not self._protected(relative):
                path = self.repo / relative
                if path.is_file():
                    files[relative] = path.read_text(encoding="utf-8")[:30_000]
        return files

    @classmethod
    def _protected(cls, path):
        normalized = path.replace("\\", "/").lstrip("./")
        return normalized.startswith(cls.PROTECTED_PREFIXES)

    def _ask_model(self, prompt, files):
        context = "\n\n".join(f"FILE: {name}\n{content}" for name, content in files.items())
        instruction = f"""You are the independent coding engineer inside the AGI project.
The primary runtime has stopped and handed control to you.
Improve the primary runtime in response to this explicit user request:
{prompt}

Return ONLY valid JSON with this exact shape:
{{"summary":"...","files":[{{"path":"relative/path.py","content":"complete new file content"}}]}}

Rules:
- You are the upgrader, not the primary runtime. The primary runtime will be started after validation.
- Inspect the supplied repository snapshot before making changes.
- Modify only primary-project code, tests, documentation, and configuration needed for the requested improvement.
- Never modify evolution/, run_upgrader.py, security/, .github/workflows/, .git/, or self_improvement/autocoder.py.
- Return complete file contents, never diffs or markdown fences.
- Prefer small, coherent changes and preserve compatible public APIs.
- Use Python standard library only unless the repository already uses a dependency.
- Do not add secrets, credentials, malware, persistence, or destructive system commands.
- Add or update tests when appropriate.

Repository snapshot:
{context}
"""
        payload = json.dumps({
            "model": self.model,
            "prompt": instruction,
            "stream": False,
            "format": "json",
        }).encode()
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
        proposal = json.loads(text)
        if not isinstance(proposal, dict):
            raise RuntimeError("coding model returned an invalid proposal")
        return proposal

    def _backup(self, changed_paths, before_contents):
        """Create a local rollback copy before modifying tracked files."""
        backup_dir = self.repo / "data" / "evolution_backups" / time.strftime("%Y%m%d-%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for relative in changed_paths:
            if relative in before_contents:
                target = backup_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(before_contents[relative], encoding="utf-8")
        return str(backup_dir)

    def _rollback(self, changed_paths, before_contents):
        """Restore tracked files from memory and remove only files created by the proposal."""
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

    def _push(self):
        result = self._run("git", "push", "origin", "main")
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git push failed")
        return True

    def improve(self, prompt):
        if not self.repo.is_dir() or not (self.repo / ".git").exists():
            raise RuntimeError(f"AGI repository not found: {self.repo}")

        status = self._run("git", "status", "--porcelain")
        if status.returncode:
            raise RuntimeError(status.stderr.strip() or "git status failed")

        # Existing uncommitted tracked changes are never touched. Untracked runtime
        # artifacts such as checkpoints and training data are allowed, but a proposal
        # may not overwrite an existing untracked path.
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
            raise RuntimeError("repository has uncommitted tracked changes; clean the worktree before self-improvement")

        before_files = set(self._files())
        before_contents = {}
        for relative in before_files:
            path = self.repo / relative
            if path.is_file() and not self._protected(relative):
                before_contents[relative] = path.read_text(encoding="utf-8")

        proposal = self._ask_model(prompt, self._snapshot())
        changes = proposal.get("files", [])
        if not isinstance(changes, list) or not changes:
            return {"ok": False, "stage": "proposal", "error": "model proposed no file changes"}
        if len(changes) > self.MAX_FILES:
            return {"ok": False, "stage": "proposal", "error": f"proposal exceeds {self.MAX_FILES} files"}

        changed_paths = []
        seen = set()
        try:
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
                if relative in untracked and relative not in before_files:
                    raise ValueError(f"proposal would overwrite untracked file: {relative}")
                seen.add(relative)
                changed_paths.append(relative)

            backup_dir = self._backup(changed_paths, before_contents)
            for relative in changed_paths:
                path = self.repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(next(item["content"] for item in changes if str(item["path"]).replace("\\", "/").lstrip("/") == relative), encoding="utf-8")

            syntax = self._run("python3", "-m", "compileall", "-q", ".")
            tests = self._run("python3", "-m", "unittest", "discover", "-s", "tests", "-v")
            if syntax.returncode or tests.returncode:
                self._rollback(changed_paths, before_contents)
                return {
                    "ok": False,
                    "stage": "validation",
                    "error": "validation failed; changes rolled back",
                    "syntax": syntax.stderr or syntax.stdout,
                    "tests": tests.stderr or tests.stdout,
                    "changed_paths": changed_paths,
                    "backup": backup_dir,
                }

            message = "autonomous improvement: " + str(proposal.get("summary", "update AGI"))[:80]
            add = self._run("git", "add", "--", *changed_paths)
            if add.returncode:
                raise RuntimeError(add.stderr.strip() or "git add failed")
            commit = self._run("git", "commit", "-m", message)
            if commit.returncode:
                raise RuntimeError(commit.stderr.strip() or "git commit failed")
            revision = self._run("git", "rev-parse", "HEAD")
            result = {
                "ok": True,
                "stage": "committed",
                "summary": proposal.get("summary", ""),
                "changed_paths": changed_paths,
                "commit": revision.stdout.strip(),
                "tests": "passed",
                "backup": backup_dir,
                "pushed": False,
            }
            if self.auto_push:
                self._push()
                result["pushed"] = True
                result["stage"] = "pushed"
            return result
        except Exception:
            self._rollback(changed_paths, before_contents)
            self._run("git", "reset", "--", *changed_paths)
            raise
