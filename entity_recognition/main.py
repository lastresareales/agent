"""Entry point for the entity recognition pipeline."""

import logging

from .config import LOG_LEVEL
from .entity_dataset import load_train_dataset, load_test_dataset, load_validation_dataset
from .knowledge_graph import KnowledgeGraph
from .model import EntityRecognitionModel

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Train and evaluate the entity recognition model."""
    logger.info("Loading datasets …")
    train_data = load_train_dataset()
    validation_data = load_validation_dataset()
    test_data = load_test_dataset()

    logger.info(
        "Dataset sizes — train: %d, validation: %d, test: %d",
        len(train_data),
        len(validation_data),
        len(test_data),
    )

    model = EntityRecognitionModel()
    model.train(list(train_data), list(validation_data))

    metrics = model.evaluate(list(test_data))
    logger.info("Evaluation results: %s", metrics)

    kg = KnowledgeGraph()
    for span in test_data:
        for entity in span.entities:
            try:
                kg.add_entity(entity)
            except OverflowError:
                logger.warning("Knowledge graph node limit reached; skipping remaining entities.")
                break

    logger.info("Knowledge graph summary: %s", kg.summary())


if __name__ == "__main__":
    run_pipeline()
