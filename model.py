# model.py

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

class EntityRecognitionModel:
    def __init__(self, model_name="dslim/bert-base-NER", num_labels=None):
        """
        Initializes the model and tokenizer.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Hardware optimization: Check if a GPU is available, otherwise default to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # If fine-tuning for a custom dataset, we might need to resize the classification head
        if num_labels:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name, 
                num_labels=num_labels, 
                ignore_mismatched_sizes=True
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
        # Push the model to the available hardware (CPU or GPU)
        self.model.to(self.device)

    def train_step(self, batch, optimizer):
        """
        Executes a single training loop step (forward pass, loss calculation, backward pass).
        """
        # 1. Put the model in training mode (enables dropout layers)
        self.model.train()
        
        # 2. Move the batch tensors to the correct hardware
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 3. Forward pass: feed data through the network
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # 4. Backward pass: calculate the gradients
        loss.backward()
        
        # 5. Update the model's weights
        optimizer.step()
        
        # 6. Clear the gradients for the next batch
        optimizer.zero_grad()
        
        return loss.item()

    def extract(self, text):
        """
        Runs inference on a new string of text to predict entities.
        """
        # Put the model in evaluation mode (disables dropout for consistent results)
        self.model.eval()
        
        # Tokenize and push to hardware
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Disable gradient tracking for inference to save memory
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_class_indices = torch.argmax(logits, dim=2)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predictions = [self.model.config.id2label[idx.item()] for idx in predicted_class_indices[0]]
        
        return list(zip(tokens, predictions))
