from entity_recognition import EntityRecognitionModel, KnowledgeGraph
from entity_recognition.utils import bio_tags_to_entities, normalize_text

tokens = normalize_text("Alice went to Paris .").split()
entities = bio_tags_to_entities(tokens, ["B-PER", "O", "O", "B-LOC", "O"])
# [{'text': 'Alice', 'label': 'PER', ...}, {'text': 'Paris', 'label': 'LOC', ...}]
