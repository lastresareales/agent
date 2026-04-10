"""Entity recognition package."""

from .entities import Entity, EntityRelation, EntitySpan
from .model import EntityRecognitionModel
from .knowledge_graph import KnowledgeGraph

__all__ = [
    "Entity",
    "EntityRelation",
    "EntitySpan",
    "EntityRecognitionModel",
    "KnowledgeGraph",
]
