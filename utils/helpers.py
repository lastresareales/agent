import re

def normalize_text(text: str) -> str:
    """
    Cleans up raw text before it goes to the tokenizer.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def bio_tags_to_entities(tokens: list, tags: list) -> list:
    """
    Takes a list of words and their raw IOB/BIO tags and merges them 
    into a clean dictionary of extracted entities.
    """
    entities = []
    # (The loop logic to match words to tags would go here)
    return entities
