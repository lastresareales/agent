# Named Entity Recognition (NER) & Knowledge Graph Pipeline

## Overview
This repository contains a modular, end-to-end Machine Learning pipeline designed to perform Named Entity Recognition (NER) and map extracted entities into a directed Knowledge Graph. 

Built as a foundational AI engineering project ahead of formal computer science coursework at Spokane Falls Community College, this architecture emphasizes Separation of Concerns, object-oriented programming, and strict memory discipline for CPU-bound environments.

## Features
* **Modular Architecture:** Code is strictly separated by function (Data Ingestion, Model Brain, Execution, Graphing).
* **Sub-word Token Alignment:** Custom PyTorch `Dataset` logic handles Hugging Face tokenizer sub-word splitting without crashing label alignments.
* **Knowledge Graph Generation:** Uses `NetworkX` to draw relationships between extracted entities and exports the mathematical structure to JSON.
* **Hardware Optimized:** Specifically configured to run inference and micro-batch training on CPU-only Linux terminal environments (e.g., Chromebook/Celeron).
* **Vite Frontend:** React interface for inspecting sample entity extraction and knowledge graph relationships.

## Frontend Quick Start
```bash
npm install
python3 -m pip install -r requirements.txt
npm run dev:api
```

In a second terminal:

```bash
npm run dev
```

The Vite app starts at the URL printed by the dev server, usually `http://localhost:5173/`.
The API runs on `http://localhost:8000/` and exposes:

* `GET /api/health`
* `POST /api/extract` with JSON body `{"text": "Linus Torvalds created Linux in Helsinki."}`
* `POST /api/chat` with JSON body `{"message": "What do you remember about Linux?"}`
* `POST /api/learn-url` with JSON body `{"url": "https://example.com/article"}`
* `GET /api/memory?query=Linux&limit=10`

Test the API with curl:

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"Linus Torvalds created Linux in Helsinki."}'
```

By default, the API uses a deterministic fallback extractor so the app responds quickly during local setup.
To opt into the BERT NER model from `model.py`, start the API with:

```bash
ENTITY_RECOGNITION_USE_MODEL=1 npm run dev:api
```

## Autonomous Memory
The API learns graph facts from every extraction by default and stores them in a local SQLite database at `memory.sqlite3`.
It also stores raw chat messages as conversation memory, so the assistant can remember preferences, project goals, names, and things that do not fit cleanly into entity relationships.
This gives the project persistent memory without retraining model weights on every message.

Disable automatic memory writes with:

```bash
ENTITY_RECOGNITION_AUTO_LEARN=0 npm run dev:api
```

Use a custom memory database path with:

```bash
ENTITY_RECOGNITION_MEMORY_DB=data/memory.sqlite3 npm run dev:api
```

Learn from a web page you provide:

```bash
curl -X POST http://localhost:8000/api/learn-url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/article"}'
```

The URL learner stores the page text as raw memory and runs entity extraction over the first chunk of readable text.

## Ollama Chat
The chat endpoint uses Ollama when a local Ollama server is available, then falls back to the built-in rule responder if Ollama is offline.

Install Ollama, pull a small model, and start the API:

```bash
ollama pull llama3.2:3b
ENTITY_RECOGNITION_USE_MODEL=1 npm run dev:api
```

Use a different Ollama model or host with:

```bash
ENTITY_RECOGNITION_OLLAMA_MODEL=mistral npm run dev:api
ENTITY_RECOGNITION_OLLAMA_URL=http://127.0.0.1:11434 npm run dev:api
```

## Python Dependencies
```bash
python3 -m pip install -r requirements.txt
```

## Python Pipeline
```bash
python main.py
```

## File Structure
```text
entity_recognition/
├── index.html             # Vite HTML entrypoint
├── package.json           # Frontend scripts and dependencies
├── vite.config.js         # Vite React configuration
├── src/                   # React frontend
├── api.py                 # FastAPI app used by the frontend
├── recognition_service.py # Shared extraction and graph response service
├── main.py                # The Orchestrator: Initializes modules and runs the pipeline
├── config.py              # The Control Center: Global variables, paths, and hyperparameters
├── entities.py            # The Blueprint: Enums and Dataclasses for strict type-safety
├── entity_dataset.py      # The Data Loader: PyTorch Dataset for JSON ingestion and token alignment
├── model.py               # The Brain: PyTorch model initialization, training step, and inference
├── knowledge_graph.py     # The Mapper: NetworkX directed graph builder and JSON exporter
├── data/                  # Standardized 80/10/10 split directories
│   ├── train/dataset.json
│   ├── validation/dataset.json
│   └── test/dataset.json
```

## Contributing

This project is open-source and welcomes contributions. Whether you are fixing a bug, adding a new feature, or optimizing the graph extraction logic, your help is appreciated.

To contribute:
1. **Fork** the repository to your own GitHub account.
2. **Clone** your fork locally.
3. **Create a new branch** for your feature or bug fix (`git checkout -b feature/your-feature-name`).
4. **Commit** your changes with clear, descriptive messages (`git commit -m "Add relation extraction logic"`).
5. **Push** the branch to your fork (`git push origin feature/your-feature-name`).
6. **Open a Pull Request (PR)** back to the `main` branch of this original repository.

Please ensure your code aligns with the existing modular structure and does not break the memory-optimized hyperparameter constraints in `config.py`.

## License

This project is licensed under the [MIT License](LICENSE). 

You are free to use, modify, and distribute this software for personal or commercial purposes, provided that the original copyright notice and permission notice are included in all copies or substantial portions of the software.
