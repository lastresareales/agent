# main.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Import our custom modules
# (Assuming config.py has TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, LEARNING_RATE, EPOCHS)
from config import TRAIN_DATA_PATH, MODEL_NAME, BATCH_SIZE, LEARNING_RATE, EPOCHS, SAVED_MODEL_PATH
from entity_dataset import NERDataset
from model import EntityRecognitionModel
from knowledge_graph import GraphBuilder

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
#        print("Starting training loop...")
 #       for epoch in range(EPOCHS):
  #          total_loss = 0
   #         for batch in dataloader:
    #            # Pass the batch to the model's train_step function
     #           loss = ner_system.train_step(batch, optimizer)
      #          total_loss += loss
      #          
    #        average_loss = total_loss / len(dataloader)
     #       print(f"Epoch {epoch + 1}/{EPOCHS} complete. Average Loss: {average_loss:.4f}")

        # 5. Save the fine-tuned model and tokenizer
        print(f"Saving model to '{SAVED_MODEL_PATH}'...")
        ner_system.model.save_pretrained(SAVED_MODEL_PATH)
        ner_system.tokenizer.save_pretrained(SAVED_MODEL_PATH)
        print("Model saved successfully.")

    except FileNotFoundError:
        print(f"\n[!] Warning: Could not find '{TRAIN_DATA_PATH}'. Skipping training loop.")
        print("[!] Please create your dataset.json to run full training.\n")

    # 6. Run an extraction test (Inference)
    # --- INFERENCE TEST ---
    print("\n--- Running Inference Test ---")
    test_sentence = "Linus Torvalds created Linux in Helsinki."
    print(f"Input: {test_sentence}")
    
    results = ner_system.extract(test_sentence)
    for entity in results:
        print(entity)
    # --- KNOWLEDGE GRAPH INTEGRATION ---
    print("\n--- Building Knowledge Graph ---")
    kg = GraphBuilder()
    
    # 1. Manually merging the token output into Nodes based on our NER results
    kg.add_entity_node("Linus Torvalds", "PER")
    kg.add_entity_node("Linux", "MISC")  # Using the model's actual (if flawed) output
    kg.add_entity_node("Helsinki", "LOC")
    
    # 2. Hardcoding the Edges (Relationships) since we lack a Relation Extraction model
    kg.add_relationship_edge("Linus Torvalds", "CREATED", "Linux")
    kg.add_relationship_edge("Linus Torvalds", "LOCATED_IN", "Helsinki")
    
    # 3. Output the results
    print(kg.get_graph_summary())
    kg.export_to_json("data/test_graph_output.json")
    # Call the extract method from our model class
    results = ner_system.extract(test_sentence)
    
    # Print the results cleanly
    for entity in results:
        print(entity)

# This ensures the script only runs if executed directly, 
# not if it is accidentally imported by another file.
if __name__ == "__main__":
    main()
