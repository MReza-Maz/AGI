from collections import deque


class WorkingMemory:
    def __init__(self, capacity=32):
        self.items=deque(maxlen=capacity)
    def add(self,item): self.items.append(item)
    def recent(self,n=None): return list(self.items if n is None else list(self.items)[-n:])
    def clear(self): self.items.clear()
    def __len__(self): return len(self.items)
