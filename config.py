# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/train/dataset.json")

# Model hyperparameters
MODEL_NAME = "dslim/bert-base-NER" # Smaller model for hardware efficiency
BATCH_SIZE = 8 # Kept small to prevent memory crashes on the Celeron processor
LEARNING_RATE = 2e-5
EPOCHS = 3
