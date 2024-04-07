from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, source, target):
        # super.__init__()
        
        self.source = source
        self.target = target
        self.length = len(source)
        
    def __getitem__(self, index):
        return self.source[index], self.target[index]
    
    def __len__(self):
        return self.length