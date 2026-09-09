class Reflection:
    def evaluate(self,goal,result=None):
        return {"goal":goal,"success":result is not None,"confidence":0.5 if result is None else 0.8,"questions":[] if result is not None else ["What evidence is missing?"]}
