# entities.py

from enum import Enum
from dataclasses import dataclass

# 1. Define the allowed entity types
# Inheriting from 'str' allows these to be compared directly to string outputs from the model
class EntityType(str, Enum):
    PERSON = "PER"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    MISCELLANEOUS = "MISC"
    OUTSIDE = "O"  # Used when a word does not belong to any entity

# 2. Define the IOB (Inside, Outside, Beginning) tags
# BERT models classify tokens using this format to handle multi-word entities (e.g., "New" = B-LOC, "York" = I-LOC)
class IOBTag(str, Enum):
    BEGIN = "B"
    INSIDE = "I"

# 3. Create a structured object to hold the final extracted data
@dataclass
class ExtractedEntity:
    """
    A simple blueprint to hold a token and its predicted label.
    Using a dataclass is cleaner than passing around raw tuples or dictionaries.
    """
    word: str
    label: str
    
    # An optional method to print the data cleanly
    def __str__(self):
        return f"{self.word:15} : {self.label}"
