from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat_service import chat
from memory_store import memory_summary, search_facts
from recognition_service import extract_text, model_available
from web_learning import learn_url


EntityLabel = Literal["PER", "ORG", "LOC", "MISC"]


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    learn: bool = True


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class LearnUrlRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048)


class EntityResponse(BaseModel):
    id: str
    text: str
    label: EntityLabel
    start: int
    end: int
    confidence: float | None = None


class GraphNode(BaseModel):
    id: str
    label: EntityLabel


class GraphEdge(BaseModel):
    source: str
    relation: str
    target: str


class ExtractResponse(BaseModel):
    text: str
    engine: str
    entities: list[EntityResponse]
    graph: dict[str, list[GraphNode] | list[GraphEdge]]
    learned_facts: list[dict[str, str | float | None]]


class MemoryFactResponse(BaseModel):
    id: int
    source: str
    relation: str
    target: str
    source_label: EntityLabel
    target_label: EntityLabel
    source_text: str
    confidence: float | None = None
    occurrences: int
    created_at: str
    updated_at: str


class MessageMemoryResponse(BaseModel):
    id: int
    role: str
    content: str
    source: str
    importance: float
    created_at: str


class ChatResponse(BaseModel):
    reply: str
    backend: str
    backend_error: str | None = None
    extraction: ExtractResponse
    memory: list[MemoryFactResponse]
    message_memory: list[MessageMemoryResponse]
    saved_message: MessageMemoryResponse


class LearnUrlResponse(BaseModel):
    url: str
    characters_read: int
    characters_learned: int
    extraction: ExtractResponse


app = FastAPI(title="Entity Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": model_available(), "memory": memory_summary()}


@app.post("/api/extract", response_model=ExtractResponse)
def extract_entities(payload: ExtractRequest):
    return extract_text(payload.text, learn=payload.learn)


@app.get("/api/memory", response_model=list[MemoryFactResponse])
def memory(query: str | None = None, limit: int = 25):
    return search_facts(query=query, limit=limit)


@app.post("/api/chat", response_model=ChatResponse)
def chat_message(payload: ChatRequest):
    return chat(payload.message)


@app.post("/api/learn-url", response_model=LearnUrlResponse)
def learn_from_url(payload: LearnUrlRequest):
    return learn_url(payload.url)
