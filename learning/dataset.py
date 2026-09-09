class ListDataset:
    def __init__(self,items): self.items=list(items)
    def __len__(self): return len(self.items)
    def __getitem__(self,index): return self.items[index]
    def batches(self,batch_size):
        for i in range(0,len(self.items),batch_size): yield self.items[i:i+batch_size]
