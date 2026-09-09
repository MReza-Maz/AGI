import unittest
from self_improvement.candidate import Proposal,Candidate
from self_improvement.sandbox import Sandbox


class SystemTests(unittest.TestCase):
    def test_high_risk_rejected(self):
        c=Candidate(Proposal("danger","test","test","high"))
        self.assertFalse(Sandbox().evaluate(c))
        self.assertEqual(c.status,"rejected-by-policy")
    def test_low_risk_passes(self):
        c=Candidate(Proposal("safe","test","test","low"))
        self.assertTrue(Sandbox().evaluate(c))


if __name__=="__main__": unittest.main()
