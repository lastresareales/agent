# model.py

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from entities import ExtractedEntity

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
        pipeline_device = 0 if self.device.type == "cuda" else -1
        self.ner_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=pipeline_device,
        )

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
        
        # 4. Clear the gradients from the previous step
        optimizer.zero_grad()
        
        # 5. Backward pass: calculate the gradients
        loss.backward()
        
        # 6. Update the model's weights
        optimizer.step()
        
        return loss.item()

    def extract(self, text):
        """
        Runs inference on a new string of text to predict entities.
        """
        grouped_entities = self.ner_pipeline(text)
        return [
            ExtractedEntity(
                word=entity["word"],
                label=entity.get("entity_group", entity.get("entity", "")),
                start=entity.get("start"),
                end=entity.get("end"),
                confidence=float(entity["score"]) if "score" in entity else None,
            )
            for entity in grouped_entities
        ]
