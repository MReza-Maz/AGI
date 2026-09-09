"""Guarded autonomous code-improvement engine using a local HTTP model.

The model proposes repository changes. This module applies them only after
syntax checks and the project's unit tests pass. Security and execution-boundary
code is protected from automatic modification.
"""
import json
import os
import subprocess
import urllib.request
from pathlib import Path


class AutonomousCoder:
    """Read the repository, ask a local coding model for changes, test, commit, and optionally push."""

    PROTECTED_PREFIXES = (
        ".git/",
        ".github/workflows/",
        "security/",
        "self_improvement/autocoder.py",
    )

    def __init__(self, repo_path="/opt/AGI", model_url=None, model=None, timeout=180, auto_push=None):
        self.repo = Path(repo_path).resolve()
        self.model_url = model_url or os.environ.get("AGI_MODEL_URL", "http://127.0.0.1:11434/api/generate")
        self.model = model or os.environ.get("AGI_MODEL", "qwen2.5-coder:7b")
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
            if relative.endswith((".py", ".json", ".md")):
                path = self.repo / relative
                if path.is_file():
                    files[relative] = path.read_text(encoding="utf-8")[:30000]
        return files

    @classmethod
    def _protected(cls, path):
        normalized = path.replace("\\", "/").lstrip("./")
        return normalized.startswith(cls.PROTECTED_PREFIXES)

    def _ask_model(self, prompt, files):
        context = "\n\n".join(f"FILE: {name}\n{content}" for name, content in files.items())
        instruction = f"""You are the coding engineer inside the AGI project.
Improve the project in response to this user request:
{prompt}

Return ONLY valid JSON with this exact shape:
{{"summary":"...","files":[{{"path":"relative/path.py","content":"complete new file content"}}]}}

Rules:
- Return complete file contents, never diffs or markdown fences.
- Prefer small, coherent changes.
- Use Python standard library only unless the repository already uses a dependency.
- Do not modify security/, .github/workflows/, .git/, or self_improvement/autocoder.py.
- Do not add secrets, credentials, malware, persistence, or destructive system commands.
- Preserve public APIs unless the request requires a compatible extension.
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
        return json.loads(text)

    def _rollback(self, changed_paths, before):
        tracked = [path for path in changed_paths if path in before]
        created = [path for path in changed_paths if path not in before]
        if tracked:
            self._run("git", "restore", "--worktree", "--", *tracked)
        if created:
            self._run("git", "clean", "-f", "--", *created)

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
        if status.stdout.strip():
            raise RuntimeError("repository has uncommitted changes; clean the worktree before self-improvement")

        before = set(self._files())
        proposal = self._ask_model(prompt, self._snapshot())
        changes = proposal.get("files", [])
        if not isinstance(changes, list) or not changes:
            return {"ok": False, "stage": "proposal", "error": "model proposed no file changes"}

        changed_paths = []
        try:
            for item in changes:
                relative = str(item.get("path", "")).replace("\\", "/").lstrip("/")
                content = item.get("content")
                if not relative or not isinstance(content, str):
                    raise ValueError("invalid model file change")
                if self._protected(relative) or ".." in Path(relative).parts:
                    raise ValueError(f"protected or invalid path: {relative}")
                path = self.repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                changed_paths.append(relative)

            syntax = self._run("python3", "-m", "compileall", "-q", ".")
            tests = self._run("python3", "-m", "unittest", "discover", "-s", "tests", "-v")
            if syntax.returncode or tests.returncode:
                self._rollback(changed_paths, before)
                return {
                    "ok": False,
                    "stage": "validation",
                    "error": "validation failed; changes rolled back",
                    "syntax": syntax.stderr or syntax.stdout,
                    "tests": tests.stderr or tests.stdout,
                    "changed_paths": changed_paths,
                }

            message = "autonomous improvement: " + str(proposal.get("summary", "update AGI"))[:80]
            commit = self._run("git", "add", "--", *changed_paths)
            if commit.returncode:
                raise RuntimeError(commit.stderr.strip() or "git add failed")
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
                "pushed": False,
            }
            if self.auto_push:
                self._push()
                result["pushed"] = True
                result["stage"] = "pushed"
            return result
        except Exception:
            # If an exception occurs before a successful commit, restore the worktree.
            current = self._run("git", "status", "--porcelain")
            if current.returncode == 0 and any(line.endswith(tuple(changed_paths)) for line in current.stdout.splitlines()):
                self._rollback(changed_paths, before)
            raise
