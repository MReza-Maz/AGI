from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory


class MemoryManager:
    def __init__(self,capacity=32):
        self.working=WorkingMemory(capacity)
        self.episodic=EpisodicMemory()
        self.semantic=SemanticMemory()
    def remember(self,event,vector=None,metadata=None):
        self.working.add(event); self.episodic.remember(event,metadata)
        if vector is not None: self.semantic.add(str(event),vector)
