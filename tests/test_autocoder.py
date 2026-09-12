import unittest

from self_improvement.autocoder import AutonomousCoder


class AutonomousCoderSafetyTests(unittest.TestCase):
    def setUp(self):
        self.coder = AutonomousCoder(repo_path=".")

    def test_protected_paths_cannot_be_modified(self):
        for path in (
            "evolution/lifecycle.py",
            "self_improvement/autocoder.py",
            "security/policy.py",
            "run_upgrader.py",
            ".github/workflows/test.yml",
        ):
            self.assertTrue(self.coder._protected(path))

    def test_primary_path_is_allowed(self):
        self.assertFalse(self.coder._protected("cognition/agent.py"))

    def test_parent_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            self.coder._validate_paths(
                [{"path": "../escape.py", "content": "print(1)"}], set()
            )

    def test_duplicate_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            self.coder._validate_paths(
                [
                    {"path": "cognition/a.py", "content": "x = 1"},
                    {"path": "cognition/a.py", "content": "x = 2"},
                ],
                set(),
            )

    def test_untracked_files_are_never_overwritten(self):
        with self.assertRaises(ValueError):
            self.coder._validate_paths(
                [{"path": "data/train.txt", "content": "replace"}],
                {"data/train.txt"},
            )

    def test_file_count_limit_is_enforced(self):
        changes = [
            {"path": f"cognition/generated_{i}.py", "content": "x = 1"}
            for i in range(self.coder.MAX_FILES + 1)
        ]
        with self.assertRaises(ValueError):
            self.coder._validate_paths(changes, set())


if __name__ == "__main__":
    unittest.main()
