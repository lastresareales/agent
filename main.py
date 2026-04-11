# main.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Import our custom modules
# (Assuming config.py has TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, LEARNING_RATE, EPOCHS)
from config import TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, LEARNING_RATE, EPOCHS
from entity_dataset import NERDataset
from model import EntityRecognitionModel

def main():
    print("Initializing the ML Pipeline...")

    # 1. Boot up the brain
    print(f"Loading model '{MODEL_NAME}' and tokenizer...")
    ner_system = EntityRecognitionModel(model_name=MODEL_NAME)

    # 2. Prepare the data 
    # Wrapping in a try-except block because the JSON file might not exist yet
    try:
        print("Loading dataset...")
        dataset = NERDataset(filepath=TRAIN_DATA_PATH, tokenizer=ner_system.tokenizer)
        
        # DataLoader handles batching, shuffling, and memory management automatically
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 3. Setup the optimizer for training
        optimizer = AdamW(ner_system.model.parameters(), lr=LEARNING_RATE)

        # 4. The Training Loop
        print("Starting training loop...")
        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in dataloader:
                # Pass the batch to the model's train_step function
                loss = ner_system.train_step(batch, optimizer)
                total_loss += loss
                
            average_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{EPOCHS} complete. Average Loss: {average_loss:.4f}")

    except FileNotFoundError:
        print(f"\n[!] Warning: Could not find '{TRAIN_DATA_PATH}'. Skipping training loop.")
        print("[!] Please create your dataset.json to run full training.\n")

    # 5. Run an extraction test (Inference)
    print("--- Running Inference Test ---")
    test_sentence = "Linus Torvalds created Linux in Helsinki."
    print(f"Input: {test_sentence}")
    
    # Call the extract method from our model class
    results = ner_system.extract(test_sentence)
    
    # Print the results cleanly
    for token, label in results:
        print(f"{token:15} : {label}")

# This ensures the script only runs if executed directly, 
# not if it is accidentally imported by another file.
if __name__ == "__main__":
    main()
