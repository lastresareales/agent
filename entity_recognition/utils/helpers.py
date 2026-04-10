"""Helper functions for text processing and entity handling."""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Compiled pattern for whitespace normalisation
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse consecutive whitespace and strip leading/trailing spaces.

    Args:
        text: Raw input string.

    Returns:
        Normalised string.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split *text* into tokens on whitespace boundaries.

    Args:
        text: Input string.

    Returns:
        List of token strings.
    """
    return normalize_text(text).split()


def char_span_to_token_span(
    tokens: list[str],
    char_start: int,
    char_end: int,
    text: str,
) -> Optional[tuple[int, int]]:
    """Convert a character-level span to a token-level span.

    Args:
        tokens: Tokenised form of *text*.
        char_start: Inclusive character start offset.
        char_end: Exclusive character end offset.
        text: The original text from which *tokens* were derived.

    Returns:
        A ``(token_start, token_end)`` tuple (exclusive end), or *None* if the
        span cannot be mapped.
    """
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    offset = 0
    for idx, token in enumerate(tokens):
        token_offset = text.find(token, offset)
        if token_offset == -1:
            continue
        token_end_offset = token_offset + len(token)

        if token_start is None and token_offset >= char_start:
            token_start = idx
        if token_end_offset <= char_end:
            token_end = idx + 1

        offset = token_end_offset

    if token_start is None or token_end is None:
        return None
    return token_start, token_end


def bio_tags_to_entities(tokens: list[str], tags: list[str]) -> list[dict]:
    """Convert a BIO-tagged sequence into a list of entity dicts.

    Args:
        tokens: List of token strings.
        tags: Parallel list of BIO tags (e.g. ``"B-PER"``, ``"I-PER"``, ``"O"``).

    Returns:
        List of dicts with keys ``text``, ``label``, ``token_start``,
        ``token_end`` (exclusive).
    """
    if len(tokens) != len(tags):
        raise ValueError("tokens and tags must have the same length.")

    entities: list[dict] = []
    current_label: Optional[str] = None
    start: Optional[int] = None

    for idx, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_label is not None:
                entities.append(
                    {
                        "text": " ".join(tokens[start:idx]),
                        "label": current_label,
                        "token_start": start,
                        "token_end": idx,
                    }
                )
            current_label = tag[2:]
            start = idx
        elif tag.startswith("I-"):
            label = tag[2:]
            if current_label != label:
                logger.warning(
                    "Unexpected I- tag '%s' at position %d; starting new entity.", tag, idx
                )
                current_label = label
                start = idx
        else:  # "O" or unknown
            if current_label is not None:
                entities.append(
                    {
                        "text": " ".join(tokens[start:idx]),
                        "label": current_label,
                        "token_start": start,
                        "token_end": idx,
                    }
                )
            current_label = None
            start = None

    if current_label is not None:
        entities.append(
            {
                "text": " ".join(tokens[start:]),
                "label": current_label,
                "token_start": start,
                "token_end": len(tokens),
            }
        )

    return entities
