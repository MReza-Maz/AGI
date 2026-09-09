from .candidate import Proposal


class Analyzer:
    def analyze(self,metrics):
        proposals=[]
        if metrics.get("reasoning_confidence",1.0)<0.7:
            proposals.append(Proposal("Improve reasoning confidence","Confidence is below target","More reliable reflection and planning","medium"))
        if metrics.get("memory_recall",1.0)<0.8:
            proposals.append(Proposal("Improve memory retrieval","Recall is below target","Better semantic retrieval","medium"))
        if not proposals:
            proposals.append(Proposal("Improve evaluation coverage","Increase measurable feedback","Broader benchmark coverage","low"))
        return proposals
