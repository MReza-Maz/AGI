import json
import os
import time


class EpisodicMemory:
    def __init__(self,path="data/episodes.jsonl"):
        self.path=path
        directory=os.path.dirname(path)
        if directory: os.makedirs(directory,exist_ok=True)
    def remember(self,event,metadata=None):
        record={"time":time.time(),"event":event,"metadata":metadata or {}}
        with open(self.path,"a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
        return record
    def recent(self,n=20):
        if not os.path.exists(self.path): return []
        with open(self.path,encoding="utf-8") as f: rows=f.readlines()[-n:]
        return [json.loads(x) for x in rows]
