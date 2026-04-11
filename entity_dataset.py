# entity_dataset.py
import json
import torch
from torch.utils.data import Dataset
from config import BATCH_SIZE

class NERDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.tokenizer = tokenizer
        # Load the dataset.json
        with open(filepath, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Logic to grab a single sentence and its labels, 
        # tokenize it, and return PyTorch tensors.
        pass
