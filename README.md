# Named Entity Recognition (NER) & Knowledge Graph Pipeline

## Overview
This repository contains a modular, end-to-end Machine Learning pipeline designed to perform Named Entity Recognition (NER) and map extracted entities into a directed Knowledge Graph. 

Built as a foundational AI engineering project ahead of formal computer science coursework at Spokane Falls Community College, this architecture emphasizes Separation of Concerns, object-oriented programming, and strict memory discipline for CPU-bound environments.

## Features
* **Modular Architecture:** Code is strictly separated by function (Data Ingestion, Model Brain, Execution, Graphing).
* **Sub-word Token Alignment:** Custom PyTorch `Dataset` logic handles Hugging Face tokenizer sub-word splitting without crashing label alignments.
* **Knowledge Graph Generation:** Uses `NetworkX` to draw relationships between extracted entities and exports the mathematical structure to JSON.
* **Hardware Optimized:** Specifically configured to run inference and micro-batch training on CPU-only Linux terminal environments (e.g., Chromebook/Celeron).

## File Structure
```text
entity_recognition/
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
