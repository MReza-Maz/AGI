class SGD:
    def __init__(self,parameters,lr=0.001): self.parameters=list(parameters); self.lr=lr
    def step(self):
        for p in self.parameters:
            if p.grad is None: continue
            def update(x,g):
                if isinstance(x,list): return [update(a,b) for a,b in zip(x,g)]
                return x-self.lr*g
            p.data=update(p.data,p.grad)
    def zero_grad(self):
        for p in self.parameters: p.grad=p._zeros_like(p.data)
