"""General-purpose utilities for the entity recognition package."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json(file_path: str | Path) -> Any:
    """Load and return the contents of a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON content (dict, list, etc.).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, file_path: str | Path, indent: int = 2) -> None:
    """Serialize *data* to a JSON file.

    Args:
        data: JSON-serialisable Python object.
        file_path: Destination file path.
        indent: Number of spaces to use for indentation.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)
    logger.debug("Saved JSON to %s", path)


def flatten(nested: list) -> list:
    """Recursively flatten a nested list.

    Args:
        nested: A (possibly nested) list.

    Returns:
        A flat list containing all leaf elements.
    """
    result: list = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def chunks(lst: list, size: int):
    """Yield successive *size*-element chunks from *lst*.

    Args:
        lst: Source list.
        size: Maximum chunk length (must be > 0).

    Yields:
        Sublists of at most *size* elements.
    """
    if size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
