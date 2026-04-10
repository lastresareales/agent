"""Entity class definitions for the entity recognition system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    """Represents a single recognized entity."""

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Entity(text={self.text!r}, label={self.label!r}, "
            f"start={self.start}, end={self.end}, confidence={self.confidence:.2f})"
        )

    def to_dict(self) -> dict:
        """Serialize entity to a dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """Deserialize entity from a dictionary."""
        return cls(
            text=data["text"],
            label=data["label"],
            start=data["start"],
            end=data["end"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EntitySpan:
    """Represents a span of text with associated entities."""

    text: str
    entities: list["Entity"] = field(default_factory=list)

    def add_entity(self, entity: "Entity") -> None:
        """Add an entity to this span."""
        self.entities.append(entity)

    def get_entities_by_label(self, label: str) -> list["Entity"]:
        """Return all entities matching the given label."""
        return [e for e in self.entities if e.label == label]


@dataclass
class EntityRelation:
    """Represents a directed relation between two entities."""

    source: Entity
    target: Entity
    relation_type: str
    confidence: float = 1.0
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize relation to a dictionary."""
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "metadata": self.metadata or {},
        }
