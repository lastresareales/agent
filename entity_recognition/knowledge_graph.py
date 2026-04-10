"""Knowledge graph construction and querying for entity recognition."""

import logging
from typing import Optional

from .config import KG_MAX_EDGES, KG_MAX_NODES
from .entities import Entity, EntityRelation

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """In-memory knowledge graph that stores entities and their relations.

    Nodes represent :class:`Entity` objects; directed edges represent
    :class:`EntityRelation` objects between them.
    """

    def __init__(
        self,
        max_nodes: int = KG_MAX_NODES,
        max_edges: int = KG_MAX_EDGES,
    ) -> None:
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self._nodes: dict[str, Entity] = {}
        self._edges: list[EntityRelation] = []

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> str:
        """Add an entity node and return its node key."""
        if len(self._nodes) >= self.max_nodes:
            raise OverflowError(f"Knowledge graph has reached the node limit ({self.max_nodes}).")
        key = self._node_key(entity)
        self._nodes[key] = entity
        logger.debug("Added entity node: %s", key)
        return key

    def get_entity(self, key: str) -> Optional[Entity]:
        """Retrieve an entity node by key, or *None* if not found."""
        return self._nodes.get(key)

    def remove_entity(self, key: str) -> None:
        """Remove an entity node and all its associated edges."""
        if key in self._nodes:
            del self._nodes[key]
            self._edges = [
                e for e in self._edges
                if self._node_key(e.source) != key and self._node_key(e.target) != key
            ]
            logger.debug("Removed entity node: %s", key)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_relation(self, relation: EntityRelation) -> None:
        """Add a directed relation between two entity nodes."""
        if len(self._edges) >= self.max_edges:
            raise OverflowError(f"Knowledge graph has reached the edge limit ({self.max_edges}).")
        self._edges.append(relation)
        logger.debug(
            "Added relation: %s -[%s]-> %s",
            relation.source.text,
            relation.relation_type,
            relation.target.text,
        )

    def get_relations(
        self,
        source_key: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> list[EntityRelation]:
        """Return relations filtered by source node key and/or relation type."""
        results = self._edges
        if source_key is not None:
            results = [e for e in results if self._node_key(e.source) == source_key]
        if relation_type is not None:
            results = [e for e in results if e.relation_type == relation_type]
        return results

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def summary(self) -> dict:
        """Return a summary of the graph statistics."""
        return {"num_nodes": self.num_nodes, "num_edges": self.num_edges}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_key(entity: Entity) -> str:
        return f"{entity.label}::{entity.text}"
