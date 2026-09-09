import json, math, os


class SemanticMemory:
    def __init__(self,path="data/semantic.json"):
        self.path=path; self.items=[]; self._load()
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path,encoding="utf-8") as f: self.items=json.load(f)
    def _save(self):
        d=os.path.dirname(self.path)
        if d: os.makedirs(d,exist_ok=True)
        with open(self.path,"w",encoding="utf-8") as f: json.dump(self.items,f,ensure_ascii=False,indent=2)
    @staticmethod
    def similarity(a,b):
        dot=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
        return dot/(na*nb) if na and nb else 0.0
    def add(self,text,vector):
        self.items.append({"text":text,"vector":list(vector)}); self._save()
    def search(self,vector,k=5):
        return sorted(self.items,key=lambda x:self.similarity(vector,x["vector"]),reverse=True)[:k]
