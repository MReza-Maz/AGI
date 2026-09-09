import json,os,time


class AuditLog:
    def __init__(self,path="data/audit.jsonl"):
        self.path=path; os.makedirs(os.path.dirname(path),exist_ok=True)
    def record(self,action,details=None):
        row={"time":time.time(),"action":action,"details":details or {}}
        with open(self.path,"a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
