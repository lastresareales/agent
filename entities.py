# entities.py
from enum import Enum

class EntityType(Enum):
    PERSON = "PER"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    MISCELLANEOUS = "MISC"
