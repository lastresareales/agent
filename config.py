# config.py

import os

# --- Path Configurations ---
# Dynamically find the absolute path to the directory where this script lives.
# This ensures the code runs correctly no matter where you launch it from your terminal.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "train", "dataset.json")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test", "dataset.json")
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, "data", "validation", "dataset.json")

# --- Model & Hyperparameters ---
# The identifier for the Hugging Face model you want to download and use.
# Using a 'base' or 'distil' model is highly recommended for CPU execution.
MODEL_NAME = "dslim/bert-base-NER"

# Training Settings
# Number of epochs: How many times the model will see the entire dataset during training.
EPOCHS = 3 

# Learning Rate: How aggressively the model updates its weights. 
# 2e-5 (0.00002) is the industry standard for fine-tuning BERT models.
LEARNING_RATE = 2e-5 

# Batch Size: How many sentences are processed at the exact same time.
# Set strictly to 2 to prevent memory overload on shared-memory CPUs.
BATCH_SIZE = 2 

# Maximum length of a tokenized sentence. Sentences longer than this will be truncated.
MAX_SEQ_LENGTH = 128

# Path where the fine-tuned model weights will be saved after training.
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "saved_model")
