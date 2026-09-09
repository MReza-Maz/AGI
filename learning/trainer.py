class Trainer:
    def __init__(self,model,optimizer,loss_fn): self.model=model; self.optimizer=optimizer; self.loss_fn=loss_fn
    def train(self,dataset,epochs=1):
        history=[]
        for _ in range(epochs):
            total=0.0
            for x,y in dataset:
                self.optimizer.zero_grad()
                loss=self.loss_fn(self.model(x),y)
                loss.backward(); self.optimizer.step(); total+=loss.data
            history.append(total/max(1,len(dataset)))
        return history
