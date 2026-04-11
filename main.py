import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

class EntityRecognition:
    def __init__(self, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
        # We start with None, to be populated when load_model is called
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        # 1. Load the Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 2. Load the Model for Token Classification (NER)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        
        # 3. Set to evaluation mode (disables dropout layers for consistent inference)
        self.model.eval()

        return self.model, self.tokenizer

    def extract_info(self, input_text):
        # Safety check to ensure the model is loaded before inferencing
        if not self.model or not self.tokenizer:
            raise ValueError("You must call load_model() before extracting info.")

        # Tokenize the text and return PyTorch tensors
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # Disable gradient tracking to save memory and speed up computation
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the logits (raw scores) and find the highest score per token
        logits = outputs.logits
        predicted_class_indices = torch.argmax(logits, dim=2)

        # Convert the token IDs back to strings so we can read them
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Convert the predicted class numbers back into human-readable labels (e.g., B-ORG, I-PER)
        predictions = [self.model.config.id2label[idx.item()] for idx in predicted_class_indices[0]]

        # Pair up the tokens with their predicted labels and return
        return list(zip(tokens, predictions))

# --- Example of how to use the class ---
if __name__ == "__main__":
    # Initialize our class
    ner_system = EntityRecognition()
    
    # Load the model and tokenizer into memory
    print("Loading model... this may take a moment.")
    ner_system.load_model()
    
    # Test text
    text = "Google's headquarters is located in Mountain View, California."
    
    # Run the extraction
    results = ner_system.extract_info(text)
    
    # Print the results neatly
    for token, label in results:
        print(f"{token:15} : {label}")
