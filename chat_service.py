import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from memory_store import save_message, search_facts, search_messages
from recognition_service import extract_text


OLLAMA_URL_ENV_VAR = "ENTITY_RECOGNITION_OLLAMA_URL"
OLLAMA_MODEL_ENV_VAR = "ENTITY_RECOGNITION_OLLAMA_MODEL"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


def fact_to_sentence(fact):
    relation = fact["relation"].replace("_", " ").lower()
    return f"{fact['source']} {relation} {fact['target']}"


def build_reply(message, extraction, memories, message_memories=None):
    learned_facts = extraction.get("learned_facts", [])
    entities = extraction.get("entities", [])
    message_memories = message_memories or []

    if learned_facts:
        learned_sentence = "; ".join(fact_to_sentence(fact) for fact in learned_facts[:3])
        if memories:
            memory_sentence = "; ".join(fact_to_sentence(fact) for fact in memories[:3])
            return f"I learned this: {learned_sentence}. I also remember: {memory_sentence}."
        return f"I learned this: {learned_sentence}."

    if message_memories:
        return f"I remember you told me: {message_memories[0]['content']}"

    if memories:
        memory_sentence = "; ".join(fact_to_sentence(fact) for fact in memories[:4])
        return f"I remember this: {memory_sentence}."

    if entities:
        entity_sentence = ", ".join(f"{entity['text']} ({entity['label']})" for entity in entities[:5])
        return f"I noticed these entities: {entity_sentence}. I do not have a strong relationship to store from that yet."

    clean_message = message.strip()
    if clean_message.endswith("?"):
        return "I do not know that yet. Tell me a fact about it, and I will remember it."

    return "I am listening. I did not find a clear entity relationship in that message yet."


def build_system_prompt(memories, message_memories, extraction):
    memory_lines = [f"- {fact_to_sentence(fact)}" for fact in memories[:12]]
    message_memory_lines = [
        f"- {memory['role']} said: {memory['content']}" for memory in message_memories[:12]
    ]
    learned_lines = [f"- {fact_to_sentence(fact)}" for fact in extraction.get("learned_facts", [])[:8]]
    entity_lines = [
        f"- {entity['text']} ({entity['label']}, confidence={entity.get('confidence')})"
        for entity in extraction.get("entities", [])[:12]
    ]

    return "\n".join(
        [
            "You are a conversational assistant with a growing memory.",
            "Speak naturally in English. Be direct, warm, and concise.",
            "Use the memory facts as context, but do not pretend uncertain facts are guaranteed.",
            "If the user teaches a new fact, acknowledge it naturally.",
            "If the user asks what you remember, answer from memory.",
            "",
            "Relevant memory:",
            "\n".join(memory_lines) if memory_lines else "- No relevant stored memory yet.",
            "",
            "Relevant raw conversation memories:",
            "\n".join(message_memory_lines) if message_memory_lines else "- No relevant raw memories yet.",
            "",
            "Facts just learned from this message:",
            "\n".join(learned_lines) if learned_lines else "- None.",
            "",
            "Entities detected in this message:",
            "\n".join(entity_lines) if entity_lines else "- None.",
        ]
    )


def ask_ollama(message, memories, message_memories, extraction):
    base_url = os.getenv(OLLAMA_URL_ENV_VAR, DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv(OLLAMA_MODEL_ENV_VAR, DEFAULT_OLLAMA_MODEL)
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": build_system_prompt(memories, message_memories, extraction)},
            {"role": "user", "content": message},
        ],
        "options": {
            "temperature": 0.7,
            "num_predict": 220,
        },
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return None, f"Ollama unavailable: {error}"

    content = body.get("message", {}).get("content", "").strip()
    if not content:
        return None, "Ollama returned an empty response"

    return content, None


def chat(message):
    save_message("user", message)
    extraction = extract_text(message, learn=True)
    query_terms = [entity["text"] for entity in extraction.get("entities", [])]
    memories = []
    seen_memory_ids = set()
    message_memories = []
    seen_message_ids = set()

    for query in query_terms or [message]:
        for fact in search_facts(query=query, limit=8):
            if fact["id"] not in seen_memory_ids:
                memories.append(fact)
                seen_memory_ids.add(fact["id"])
        for memory in search_messages(query=query, limit=8):
            if memory["id"] not in seen_message_ids:
                message_memories.append(memory)
                seen_message_ids.add(memory["id"])
        if len(memories) >= 8:
            break

    if not message_memories:
        message_memories = search_messages(limit=8)

    ollama_reply, ollama_error = ask_ollama(message, memories, message_memories, extraction)
    reply = ollama_reply or build_reply(message, extraction, memories, message_memories)
    assistant_memory = save_message("assistant", reply, importance=0.4)

    return {
        "reply": reply,
        "backend": "ollama" if ollama_reply else "rules",
        "backend_error": ollama_error,
        "extraction": extraction,
        "memory": memories,
        "message_memory": message_memories,
        "saved_message": assistant_memory,
    }
