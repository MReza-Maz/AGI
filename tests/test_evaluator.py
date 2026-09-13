import tempfile
import unittest
from pathlib import Path

from evaluator import Evaluator, EvaluationError


class EvaluatorTests(unittest.TestCase):
    def test_syntax_check(self):
        Evaluator().syntax_check("print('ok')")

    def test_invalid_syntax_rejected(self):
        with self.assertRaises(SyntaxError):
            Evaluator().syntax_check("def broken(:")

    def test_output_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.py"
            candidate = root / "candidate.py"
            baseline.write_text("print('A')\n", encoding="utf-8")
            candidate.write_text("print('B')\n", encoding="utf-8")
            with self.assertRaises(EvaluationError):
                Evaluator(5, 1).evaluate(candidate, baseline)


if __name__ == "__main__":
    unittest.main()
