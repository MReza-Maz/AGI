from .candidate import Candidate
from .sandbox import Sandbox
from .analyzer import Analyzer
from security.approval import ApprovalGate
from security.audit import AuditLog
from security.checkpoint import CheckpointManager


class SelfImprovementController:
    def __init__(self,require_approval=True):
        self.analyzer=Analyzer(); self.sandbox=Sandbox(); self.approval=ApprovalGate(require_approval)
        self.audit=AuditLog(); self.checkpoints=CheckpointManager()
    def propose(self,metrics):
        proposals=self.analyzer.analyze(metrics)
        candidates=[]
        for p in proposals:
            c=Candidate(p); self.audit.record("proposal",{"title":p.title})
            if self.sandbox.evaluate(c) and self.approval.approve(c):
                self.checkpoints.save({"proposal":p.title},"pre-deploy")
                c.status="ready-for-deploy"
                self.audit.record("approved",{"title":p.title})
            candidates.append(c)
        return candidates
