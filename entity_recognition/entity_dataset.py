"""Dataset loading and processing for entity recognition."""

import json
import logging
from pathlib import Path
from typing import Iterator

from .config import TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH
from .entities import Entity, EntitySpan

logger = logging.getLogger(__name__)


class EntityDataset:
    """Loads and provides access to entity recognition examples."""

    def __init__(self, data_path: str) -> None:
        self.data_path = Path(data_path)
        self._examples: list[EntitySpan] = []
        self._load()

    def _load(self) -> None:
        """Load examples from the JSON dataset file."""
        if not self.data_path.exists():
            logger.warning("Dataset file not found: %s", self.data_path)
            return

        with self.data_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        for record in raw.get("examples", []):
            span = EntitySpan(text=record["text"])
            for ent_data in record.get("entities", []):
                span.add_entity(Entity.from_dict(ent_data))
            self._examples.append(span)

        logger.info("Loaded %d examples from %s", len(self._examples), self.data_path)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> EntitySpan:
        return self._examples[index]

    def __iter__(self) -> Iterator[EntitySpan]:
        return iter(self._examples)


def load_train_dataset() -> EntityDataset:
    """Load the training split."""
    return EntityDataset(TRAIN_DATA_PATH)


def load_test_dataset() -> EntityDataset:
    """Load the test split."""
    return EntityDataset(TEST_DATA_PATH)


def load_validation_dataset() -> EntityDataset:
    """Load the validation split."""
    return EntityDataset(VALIDATION_DATA_PATH)
