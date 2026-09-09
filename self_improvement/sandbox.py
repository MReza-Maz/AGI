class Sandbox:
    """Policy boundary: this version evaluates metadata only and never executes candidate code."""
    def evaluate(self,candidate):
        if candidate.proposal.risk.lower() in {"critical","high"}:
            candidate.status="rejected-by-policy"
            return False
        candidate.status="sandbox-passed"
        return True
