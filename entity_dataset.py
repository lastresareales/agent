import pandas as pd

class EntityDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def __len__(self):
        return len(self.data)
