# RAG Document Q&A Constitution

## Core Principles

### I. Specification Before Code
Every piece of behavior traces back to a requirement in `specs/001-rag-document-qa/spec.md`.
When behavior needs to change, the spec is updated first, then the plan and tasks, then the code.
No feature is implemented that is not in the spec.

### II. Local-First, No Paid Services
The system runs fully offline once models are downloaded: Sentence Transformers for embeddings,
Ollama for generation, ChromaDB and SQLite for storage. No API keys are required to run or test it.

### III. Test-First for Core Logic
Chunking, extraction, retrieval, prompt building, and error handling are covered by automated tests.
Tests run offline and fast by replacing the embedding model and the LLM with deterministic fakes.
The suite has at least 6 meaningful tests, each tied to a requirement or edge case.

### IV. Grounded, Cited Answers
Answers are generated only from retrieved document passages. Every answer cites its sources.
When no relevant passage is found, the system says so instead of guessing.

### V. Explicit Failures
Every failure returns a clear, actionable message. Nothing fails silently: document processing
records its status and, on failure, the reason. External dependencies (embedding model, LLM)
failing never crashes the service.

### VI. Simplicity and Explainability
Every dependency, abstraction, and parameter must be explainable by the author in an interview.
Tunable values (chunk size, overlap, top-k, threshold, model names, limits) live in configuration,
not in code. Prefer the simplest design that meets the spec.

## Technical Constraints

- Python 3.11+.
- Containerized: `Dockerfile` + `docker-compose.yml` bring up the full system.
- Supported input formats: PDF, TXT, Markdown.
- Key implementation decisions are documented with their reasoning in `research.md` and `README.md`.
- AI tool usage is documented honestly in `AI_USAGE.md`.

## Development Workflow

1. `spec.md` - what and why (no implementation details).
2. `plan.md`, `research.md`, `data-model.md`, `contracts/` - how.
3. `tasks.md` - ordered, testable tasks.
4. Implement task by task; each task ends with passing tests.

## Governance

This constitution overrides ad-hoc decisions. Amendments are made by editing this file, with the
reason recorded in the version history below.

**Version**: 1.0.0 | **Ratified**: 2026-10-13 | **Last Amended**: 2026-10-13
