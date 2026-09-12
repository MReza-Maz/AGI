import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolution.lifecycle import EvolutionLifecycle, start_primary


class EvolutionLifecycleTests(unittest.TestCase):
    def test_request_upgrade_starts_independent_upgrader(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "run_upgrader.py").write_text("print('upgrader')\n", encoding="utf-8")
            lifecycle = EvolutionLifecycle(repo, "run_browser_gateway.py")

            with patch("evolution.lifecycle.subprocess.Popen") as popen:
                result = lifecycle.request_upgrade("upgrade yourself")

            self.assertTrue(result["accepted"])
            command = popen.call_args.args[0]
            self.assertEqual(command[1], str(repo / "run_upgrader.py"))
            self.assertIn("--prompt", command)
            self.assertIn("upgrade yourself", command)
            self.assertTrue(lifecycle.upgrade_requested)

    def test_second_upgrade_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "run_upgrader.py").write_text("print('upgrader')\n", encoding="utf-8")
            lifecycle = EvolutionLifecycle(repo)

            with patch("evolution.lifecycle.subprocess.Popen"):
                self.assertTrue(lifecycle.request_upgrade("first")["accepted"])
                result = lifecycle.request_upgrade("second")

            self.assertFalse(result["accepted"])
            self.assertIn("already", result["reason"])

    def test_start_primary_uses_a_detached_process(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            script = repo / "run_browser_gateway.py"
            script.write_text("print('primary')\n", encoding="utf-8")

            with patch("evolution.lifecycle.subprocess.Popen") as popen:
                start_primary(repo, "run_browser_gateway.py")

            kwargs = popen.call_args.kwargs
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(popen.call_args.args[0][1], str(script))


if __name__ == "__main__":
    unittest.main()
