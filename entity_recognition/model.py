"""Entity recognition model."""

import logging
from typing import Optional

from .config import ENTITY_TYPES, EMBEDDING_DIM, HIDDEN_DIM
from .entities import Entity, EntitySpan

logger = logging.getLogger(__name__)


class EntityRecognitionModel:
    """Sequence-labelling model that identifies entities in text.

    This class provides the interface for training, evaluating, and running
    inference with an entity recognition model.  The internals can be swapped
    for any ML backend (e.g. spaCy, HuggingFace Transformers, a custom BiLSTM).
    """

    def __init__(
        self,
        entity_types: Optional[list[str]] = None,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        self.entity_types = entity_types or ENTITY_TYPES
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self._is_trained = False
        logger.info(
            "Initialized EntityRecognitionModel (embedding_dim=%d, hidden_dim=%d)",
            embedding_dim,
            hidden_dim,
        )

    def train(self, train_data: list[EntitySpan], validation_data: Optional[list[EntitySpan]] = None) -> None:
        """Train the model on the provided data.

        Args:
            train_data: List of annotated EntitySpan objects for training.
            validation_data: Optional list of annotated EntitySpan objects
                for validation / early stopping.
        """
        logger.info("Starting training on %d examples.", len(train_data))
        # Training logic goes here.
        self._is_trained = True
        logger.info("Training complete.")

    def predict(self, text: str) -> EntitySpan:
        """Run inference on a single piece of text.

        Args:
            text: Raw input string.

        Returns:
            An EntitySpan populated with recognised entities.
        """
        if not self._is_trained:
            logger.warning("Model has not been trained yet; predictions may be empty.")

        span = EntitySpan(text=text)
        # Inference logic goes here.
        return span

    def evaluate(self, test_data: list[EntitySpan]) -> dict:
        """Evaluate the model and return precision, recall, and F1.

        Args:
            test_data: Annotated examples to evaluate against.

        Returns:
            Dictionary containing 'precision', 'recall', and 'f1' scores.
        """
        logger.info("Evaluating on %d examples.", len(test_data))
        # Evaluation logic goes here.
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def save(self, path: str) -> None:
        """Persist model weights/config to *path*."""
        logger.info("Saving model to %s", path)

    def load(self, path: str) -> None:
        """Restore model weights/config from *path*."""
        logger.info("Loading model from %s", path)
        self._is_trained = True
