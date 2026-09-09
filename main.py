import json
from core.brain import Brain
from core.tensor import Tensor
from memory.manager import MemoryManager
from reasoning.engine import ReasoningEngine
from self_improvement.controller import SelfImprovementController


def load_config(path="config.json"):
    with open(path,encoding="utf-8") as f: return json.load(f)


def main():
    cfg=load_config()
    brain_cfg=cfg["brain"]
    brain=Brain(brain_cfg["input_size"],brain_cfg["hidden_size"],brain_cfg["output_size"])
    memory=MemoryManager(cfg["memory"]["working_capacity"])
    reasoning=ReasoningEngine()
    controller=SelfImprovementController(cfg["security"]["require_human_approval"])

    signal=Tensor([[0.1]*brain_cfg["input_size"]])
    thought=brain.think(signal)
    memory.remember("Initial cognitive observation",thought.data[0])
    result=reasoning.reason("Build a more capable and reliable cognitive system")
    print("AGI research runtime started")
    print("Plan steps:",len(result["plan"]))
    print("Thought shape:",thought.shape)
    proposals=controller.propose({"reasoning_confidence":result["reflection"]["confidence"],"memory_recall":1.0})
    print("Improvement candidates:",len(proposals))


if __name__=="__main__": main()
