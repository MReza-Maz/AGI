class ApprovalGate:
    def __init__(self,required=True): self.required=required
    def approve(self,candidate):
        if not self.required:
            candidate.approved=True; candidate.status="approved"
            return True
        answer=input(f"Approve candidate '{candidate.proposal.title}'? [y/N]: ").strip().lower()
        candidate.approved=answer in {"y","yes"}
        candidate.status="approved" if candidate.approved else "rejected"
        return candidate.approved
