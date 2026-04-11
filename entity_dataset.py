# entity_dataset.py

import json
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        """
        Initializes the dataset by loading the JSON file.
        Expects JSON format: [{"tokens": ["The", "dog"], "ner_tags": [0, 0]}, ...]
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the raw data into memory
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        """Returns the total number of sentences in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Grabs a single sentence by its index, tokenizes it, 
        and aligns the NER labels to the new sub-word tokens.
        """
        item = self.data[idx]
        words = item["tokens"]
        original_labels = item["ner_tags"]

        # Tokenize the words. is_split_into_words=True tells the tokenizer 
        # that we are feeding it a list of words, not a single string.
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove the extra batch dimension PyTorch adds by default
        item_tensors = {key: val.squeeze(0) for key, val in encoding.items()}

        # Get the word IDs to align our labels
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        
        # The Label Alignment Rule
        for word_id in word_ids:
            if word_id is None:
                # Special tokens (like [CLS] and [SEP]) get a label of -100 
                # so PyTorch ignores them during training.
                aligned_labels.append(-100)
            else:
                # Assign the original label to the first piece of a split word,
                # and -100 to any subsequent sub-word pieces.
                aligned_labels.append(original_labels[word_id])

        item_tensors["labels"] = torch.tensor(aligned_labels, dtype=torch.long)

        return item_tensors

