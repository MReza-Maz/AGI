import json,os,time


class CheckpointManager:
    def __init__(self,path="data/checkpoints.jsonl"):
        self.path=path; os.makedirs(os.path.dirname(path),exist_ok=True)
    def save(self,state,label="checkpoint"):
        row={"time":time.time(),"label":label,"state":state}
        with open(self.path,"a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
        return row
