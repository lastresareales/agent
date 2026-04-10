"""Configuration settings for the entity recognition system."""

# Data paths (resolved relative to this config file so imports work from any cwd)
from pathlib import Path as _Path

_DATA_DIR = _Path(__file__).parent / "data"
TRAIN_DATA_PATH = str(_DATA_DIR / "train" / "dataset.json")
TEST_DATA_PATH = str(_DATA_DIR / "test" / "dataset.json")
VALIDATION_DATA_PATH = str(_DATA_DIR / "validation" / "dataset.json")

# Model settings
MODEL_NAME = "entity-recognition-model"
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Training settings
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 5  # Early stopping patience

# Entity types
ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "EVENT",
    "PRODUCT",
    "OTHER",
]

# Knowledge graph settings
KG_MAX_NODES = 10_000
KG_MAX_EDGES = 50_000

# Logging
LOG_LEVEL = "INFO"
