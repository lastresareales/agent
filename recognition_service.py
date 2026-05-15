import os
import re
from functools import lru_cache

from memory_store import learn_from_extraction


LABELS = {"PER", "ORG", "LOC", "MISC"}
MODEL_ENV_VAR = "ENTITY_RECOGNITION_USE_MODEL"
LEARN_ENV_VAR = "ENTITY_RECOGNITION_AUTO_LEARN"

FALLBACK_PATTERNS = [
    (r"\bLinus Torvalds\b", "PER"),
    (r"\bSundar Pichai\b", "PER"),
    (r"\bAda Lovelace\b", "PER"),
    (r"\bGrace Hopper\b", "PER"),
    (r"\bGoogle\b", "ORG"),
    (r"\bMicrosoft\b", "ORG"),
    (r"\bOpenAI\b", "ORG"),
    (r"\bLinux\b", "MISC"),
    (r"\bPython\b", "MISC"),
    (r"\bHelsinki\b", "LOC"),
    (r"\bMountain View\b", "LOC"),
    (r"\bSeattle\b", "LOC"),
    (r"\bSan Francisco\b", "LOC"),
]


@lru_cache(maxsize=1)
def get_model():
    if not use_model_enabled():
        return None

    try:
        from model import EntityRecognitionModel

        return EntityRecognitionModel()
    except Exception as error:
        print(f"Falling back to pattern extractor; model failed to load: {error}")
        return None


def use_model_enabled():
    return os.getenv(MODEL_ENV_VAR, "").lower() in {"1", "true", "yes"}


def model_available():
    return use_model_enabled() and get_model() is not None


def auto_learn_enabled():
    return os.getenv(LEARN_ENV_VAR, "1").lower() in {"1", "true", "yes"}


def normalize_label(label):
    clean_label = label.split("-", 1)[-1].upper()
    return clean_label if clean_label in LABELS else None


def find_span(text, term, cursor):
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    match = pattern.search(text, cursor)
    if match is None:
        match = pattern.search(text)
    if match is None:
        return None
    return match.start(), match.end()


def merge_model_tokens(text, token_entities):
    entities = []
    active_words = []
    active_label = None
    cursor = 0

    def flush():
        nonlocal active_words, active_label, cursor
        if not active_words or active_label is None:
            active_words = []
            active_label = None
            return

        entity_text = " ".join(active_words)
        span = find_span(text, entity_text, cursor)
        if span is None:
            active_words = []
            active_label = None
            return

        start, end = span
        entities.append(
            {
                "id": f"{entity_text}-{start}",
                "text": text[start:end],
                "label": active_label,
                "start": start,
                "end": end,
                "confidence": None,
            }
        )
        cursor = end
        active_words = []
        active_label = None

    for token in token_entities:
        label = normalize_label(token.label)
        if label is None:
            flush()
            continue

        starts_entity = token.label.startswith("B-") or label != active_label
        if starts_entity:
            flush()
            active_words = [token.word]
            active_label = label
        else:
            active_words.append(token.word)

    flush()
    return entities


def model_entities_to_response(text, model_entities):
    entities = []
    cursor = 0

    for index, model_entity in enumerate(model_entities):
        label = normalize_label(model_entity.label)
        if label is None:
            continue

        start = model_entity.start
        end = model_entity.end

        if start is None or end is None:
            span = find_span(text, model_entity.word, cursor)
            if span is None:
                continue
            start, end = span

        entity_text = text[start:end]
        entities.append(
            {
                "id": f"{entity_text}-{start}",
                "text": entity_text,
                "label": label,
                "start": start,
                "end": end,
                "confidence": model_entity.confidence,
            }
        )
        cursor = end

    entities.sort(key=lambda entity: (entity["start"], entity["end"]))
    return merge_adjacent_entities(text, entities)


def merge_adjacent_entities(text, entities):
    if not entities:
        return []

    merged = [entities[0].copy()]
    for entity in entities[1:]:
        previous = merged[-1]
        gap = text[previous["end"] : entity["start"]]
        contiguous_fragment = gap == ""
        same_entity_phrase = previous["label"] == entity["label"] and gap.strip() == ""

        if same_entity_phrase or contiguous_fragment:
            merged_label = previous["label"]
            if previous["label"] != entity["label"]:
                merged_label = infer_contiguous_label(text, previous, entity)

            previous["text"] = text[previous["start"] : entity["end"]]
            previous["end"] = entity["end"]
            previous["id"] = f"{previous['text']}-{previous['start']}"
            previous["label"] = merged_label
            if previous["confidence"] is not None and entity["confidence"] is not None:
                previous["confidence"] = min(previous["confidence"], entity["confidence"])
            else:
                previous["confidence"] = previous["confidence"] or entity["confidence"]
        else:
            merged.append(entity)

    return merged


def infer_contiguous_label(text, previous, entity):
    context_before = text[max(0, previous["start"] - 24) : previous["start"]].lower()
    if re.search(r"\b(from|in|to|near|at|visited|based in|headquartered in)\s*$", context_before):
        return "LOC"
    return previous["label"]


def fallback_extract(text):
    entities = []
    for pattern, label in FALLBACK_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(
                {
                    "id": f"{match.group(0)}-{match.start()}",
                    "text": match.group(0),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 1.0,
                }
            )

    entities.sort(key=lambda entity: (entity["start"], entity["end"]))
    deduped = []
    occupied = set()
    for entity in entities:
        span = (entity["start"], entity["end"])
        if span not in occupied:
            deduped.append(entity)
            occupied.add(span)
    return deduped


def add_edge(edges, seen_edges, source, relation, target):
    key = (source, relation, target)
    if key not in seen_edges:
        edges.append({"source": source, "relation": relation, "target": target})
        seen_edges.add(key)


def has_edge_pair(edges, source, target):
    return any(edge["source"] == source and edge["target"] == target for edge in edges)


def relation_between(text, source, target):
    if source["end"] <= target["start"]:
        between = text[source["end"] : target["start"]].lower()
    elif target["end"] <= source["start"]:
        between = text[target["end"] : source["start"]].lower()
    else:
        return None

    if source["label"] == "PER" and target["label"] in {"ORG", "MISC"}:
        if re.search(r"\b(found(?:ed|er)?|co-founded)\b", between):
            return "FOUNDED"
        if re.search(r"\b(created|built|developed|invented)\b", between):
            return "CREATED"
        if re.search(r"\b(leads|runs|heads|ceo of)\b", between):
            return "LEADS"
        if re.search(r"\b(works at|works for|joined)\b", between):
            return "WORKS_AT"

    if source["label"] == "ORG" and target["label"] == "LOC":
        if re.search(r"\b(based in|headquartered in|from|in)\b", between):
            return "LOCATED_IN"

    if source["label"] == "PER" and target["label"] == "LOC":
        if re.search(r"\b(from|in|visited|born in|lives in|based in)\b", between):
            return "MENTIONED_IN"

    return None


def build_graph(entities, text=""):
    unique_entities = list({entity["text"]: entity for entity in entities}.values())
    nodes = [{"id": entity["text"], "label": entity["label"]} for entity in unique_entities]
    edges = []
    seen_edges = set()

    if text:
        ordered_entities = sorted(unique_entities, key=lambda entity: (entity["start"], entity["end"]))
        for source_index, source in enumerate(ordered_entities):
            for target in ordered_entities[source_index + 1 :]:
                relation = relation_between(text, source, target)
                if relation is not None:
                    add_edge(edges, seen_edges, source["text"], relation, target["text"])

    people = [entity for entity in unique_entities if entity["label"] == "PER"]
    organizations = [entity for entity in unique_entities if entity["label"] == "ORG"]
    locations = [entity for entity in unique_entities if entity["label"] == "LOC"]
    misc = [entity for entity in unique_entities if entity["label"] == "MISC"]

    for person in people:
        for target in misc[:1]:
            if not has_edge_pair(edges, person["text"], target["text"]):
                add_edge(edges, seen_edges, person["text"], "RELATED_TO", target["text"])
        for organization in organizations[:1]:
            if not has_edge_pair(edges, person["text"], organization["text"]):
                add_edge(edges, seen_edges, person["text"], "ASSOCIATED_WITH", organization["text"])
        for location in locations[:1]:
            if not has_edge_pair(edges, person["text"], location["text"]):
                add_edge(edges, seen_edges, person["text"], "MENTIONED_IN", location["text"])

    return {"nodes": nodes, "edges": edges}


def extract_text(text, learn=None):
    model = get_model()
    if model is not None:
        entities = model_entities_to_response(text, model.extract(text))
        engine = "bert"
    else:
        entities = fallback_extract(text)
        engine = "fallback"

    result = {
        "text": text,
        "engine": engine,
        "entities": entities,
        "graph": build_graph(entities, text),
    }

    should_learn = auto_learn_enabled() if learn is None else learn
    result["learned_facts"] = learn_from_extraction(result) if should_learn else []
    return result
