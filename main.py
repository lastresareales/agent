# main.py
from config import TRAIN_DATA_PATH, MODEL_NAME
from model import EntityRecognitionModel
from entity_dataset import NERDataset
from knowledge_graph import GraphBuilder

def main():
    print("Starting pipeline...")
    
    # 1. Initialize Model
    model = EntityRecognitionModel(MODEL_NAME)
    
    # 2. Load Data
    dataset = NERDataset(TRAIN_DATA_PATH, model.tokenizer)
    
    # 3. Extract Entities
    sample_text = "Sundar Pichai works at Google."
    entities = model.extract(sample_text)
    
    # 4. Build Graph
    graph = GraphBuilder()
    # Logic to pass entities into the graph builder
    
if __name__ == "__main__":
    main()
