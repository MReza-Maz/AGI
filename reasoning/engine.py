from .planner import Planner
from .reflection import Reflection


class ReasoningEngine:
    def __init__(self):
        self.planner=Planner(); self.reflection=Reflection()
    def reason(self,goal,result=None):
        plan=self.planner.plan(goal)
        review=self.reflection.evaluate(goal,result)
        return {"goal":goal,"plan":plan,"reflection":review}
