# Implementation Plan: RAG Document Q&A

**Branch**: `001-rag-document-qa` | **Date**: 2026-10-13 | **Spec**: [spec.md](./spec.md)

## Summary

A FastAPI service that ingests PDF/TXT/Markdown documents, splits them into overlapping
token-sized chunks, embeds them locally with Sentence Transformers, stores vectors in ChromaDB and
metadata in SQLite, and answers questions by retrieving the most relevant chunks and prompting a
local Ollama model to answer with citations.

## Technical Context

| Item | Choice |
|------|--------|
| Language | Python 3.12 (project requires 3.11+) |
| Web framework | FastAPI + Uvicorn |
| Metadata storage | SQLite via SQLAlchemy 2.0 |
| Vector store | ChromaDB (embedded, persistent, cosine distance) |
| Embeddings | Sentence Transformers `BAAI/bge-small-en-v1.5` (384 dims, 512-token max input) |
| LLM | Ollama, default `llama3.2:3b`, called over its REST API with `httpx` |
| PDF extraction | `pypdf` |
| Background processing | FastAPI `BackgroundTasks` |
| Testing | `pytest`, FastAPI `TestClient`, fake embedder + fake LLM |
| Deployment | Docker + docker-compose (`app` + `ollama` services) |
| Constraints | Fully local, no API keys; 20 MB upload limit enforced before the body is read; 1000-page PDF limit; chunk <= 500 model tokens |
| Scale | Single user, tens of documents, thousands of chunks |

## Constitution Check

| Principle | How this plan complies |
|-----------|------------------------|
| I. Spec before code | Every stage lists the spec requirements it implements |
| II. Local-first | Sentence Transformers + Ollama + Chroma + SQLite; no external APIs |
| III. Test-first | Every stage from 2 onward ends with passing tests; fakes keep tests offline |
| IV. Grounded answers | Stage 8 prompt rules + Stage 9 no-information path skip the LLM entirely |
| V. Explicit failures | `errors.py` maps every domain error to a clear HTTP response; `failed` status stores reason |
| VI. Simplicity | Embedded Chroma (no extra container), SQLite, no task queue, no frameworks beyond FastAPI |

## Project Structure

```
.specify/memory/constitution.md
specs/001-rag-document-qa/        spec.md, plan.md, research.md, data-model.md,
                                  contracts/openapi.yaml, quickstart.md, tasks.md
app/
  main.py  config.py  errors.py  schemas.py  dependencies.py
  api/       documents.py  ask.py
  db/        database.py  models.py
  services/  extraction.py  chunking.py  embeddings.py  vector_store.py  ingestion.py
             retrieval.py  prompts.py  llm.py  qa.py
tests/
  conftest.py  fakes.py  fixtures/
  test_extraction.py  test_chunking.py  test_vector_store.py  test_documents_api.py
  test_retrieval.py  test_prompts.py  test_llm.py  test_ask.py  test_error_handling.py
Dockerfile  docker-compose.yml  requirements.txt  requirements-dev.txt  pyproject.toml
README.md  AI_USAGE.md
```

## Stages

Each stage is small enough to review in one sitting. A stage is finished only when its
"Done when" list is fully true. Then it is committed, and the next stage starts.

**Working loop for every stage**

1. Re-read the spec requirements the stage covers.
2. Agree on the design of the stage (functions, inputs/outputs, edge cases) before coding.
3. Write the tests and the code.
4. Run the full test suite.
5. Review: you read every new line and ask about anything unclear.
6. Commit with a message naming the stage.

---

### Stage 0 - Design artifacts

**Goal**: Finish the "how" documents so implementation has no open design questions.

- **Delivers**: `research.md` (5 key decisions + alternatives), `data-model.md` (tables, fields,
  status transitions), `contracts/openapi.yaml` (requests, responses, error codes),
  `quickstart.md` (manual curl scenarios).
- **Done when**: You have reviewed and approved all four documents.

### Stage 1 - Foundation

**Goal**: A running, testable skeleton.

- **Spec**: Constitution technical constraints.
- **Files**: `config.py`, `main.py`, `errors.py`, `db/database.py`, `tests/conftest.py`.
- **Delivers**: Virtual environment, installed dependencies, settings with validation, app factory,
  error-to-HTTP mapping, DB engine/session, test fixture using a temporary data directory.
- **Done when**: `pytest` passes; `uvicorn app.main:app` serves `/health` and `/docs`.

### Stage 2 - Data model

**Goal**: Persist documents and chunks.

- **Spec**: Key Entities, FR-003, FR-007, FR-009.
- **Files**: `db/models.py`, `schemas.py`.
- **Delivers**: `Document` and `Chunk` tables, `DocumentStatus` enum, cascade delete from document
  to chunks, Pydantic response schemas.
- **Done when**: Tests confirm a document with chunks can be created, read, and deleted with its
  chunks removed.

### Stage 3 - Text extraction

**Goal**: Turn any supported file into text, page by page.

- **Spec**: FR-005, edge cases 1 and 5.
- **Files**: `services/extraction.py`, `tests/fixtures/`, `tests/test_extraction.py`.
- **Delivers**: Two entry points. A lightweight `validate` (open PDF, page count, stop at the first
  page with text) used by the upload request, and a full `extract` (all pages, with page numbers)
  used by the background task. TXT/MD decoding (UTF-8 with fallback). Errors for unsupported types,
  unreadable files, too many pages, and files with no extractable text.
- **Done when**: Tests pass for PDF (multi-page), TXT, MD, non-UTF-8 text, empty file, image-only
  PDF, corrupt PDF, page limit, unsupported extension, and that `validate` stops after the first
  page with text.

### Stage 4 - Chunking

**Goal**: Split text into overlapping, token-sized, paragraph-aware chunks.

- **Spec**: FR-006, FR-007.
- **Files**: `services/chunking.py`, `tests/test_chunking.py`.
- **Delivers**: Paragraph packing across the whole document (page breaks are not boundaries) up to
  `chunk_size` tokens, overlap of `chunk_overlap` tokens, sentence-level splitting for oversized
  paragraphs, `page_start` / `page_end` and token count per chunk.
- **Done when**: Tests prove no chunk exceeds the limit, consecutive chunks overlap, paragraphs are
  not split when they fit, text continuing across a page break lands in one chunk with the correct
  page range, and empty input yields no chunks.

### Stage 5 - Embeddings and vector store

**Goal**: Embed chunks and search them by similarity.

- **Spec**: FR-008, FR-010, FR-011, FR-012.
- **Files**: `services/embeddings.py`, `services/vector_store.py`, `tests/fakes.py`,
  `tests/test_vector_store.py`.
- **Delivers**: `Embedder` interface with a Sentence Transformers implementation (batched,
  normalized, query prefix for bge), loaded once at startup with fail-fast checks on model loading,
  `embedding_dim`, and `chunk_size <= max_seq_length - 2`; Chroma wrapper with add, search (top-k, document filter),
  and delete-by-document; similarity score = 1 - cosine distance.
- **Done when**: Tests with the fake embedder confirm ranking order, top-k limit, document filter,
  and deletion. One manual check embeds real text with the real model.

### Stage 6 - Ingestion pipeline and document API

**Goal**: Upload a document and have it processed in the background.

- **Spec**: User Stories 1 and 3, FR-001 to FR-004a, FR-009, edge cases 1, 2, 5, 7.
- **Files**: `services/ingestion.py`, `api/documents.py`, `dependencies.py`,
  `tests/test_documents_api.py`.
- **Delivers**: `POST /documents` (returns `202`), `GET /documents`, `GET /documents/{id}`,
  `DELETE /documents/{id}`; upload-size middleware rejecting oversized bodies before they are read,
  plus an exact size check in the handler; lightweight validation in a worker thread so empty,
  unreadable, or too-long files are rejected immediately (research 6.1); background
  extract + chunk + embed pipeline with status `pending -> processing -> completed | failed`;
  startup marks interrupted documents `failed`; delete cleans SQLite, Chroma, and the stored file,
  including mid-processing.
- **Done when**: Tests cover upload-to-completed, list, metadata, delete, each rejection case, and
  a simulated embedding failure ending in `failed` with a reason.

### Stage 7 - Retrieval

**Goal**: Find the right chunks for a question.

- **Spec**: User Story 4, FR-010 to FR-012, edge case 4 and "document not ready".
- **Files**: `services/retrieval.py`, `tests/test_retrieval.py`.
- **Delivers**: Query embedding, top-k search, threshold filtering, validation of `document_ids`
  (unknown -> error, not completed -> error), default search over completed documents only,
  results enriched with document name and page.
- **Done when**: Tests confirm the passage holding a known fact ranks first, filters are honored,
  and invalid ids produce the correct errors.

### Stage 8 - Prompt and LLM client

**Goal**: Build a grounded prompt and call the model safely.

- **Spec**: FR-013, FR-014, edge case 6.
- **Files**: `services/prompts.py`, `services/llm.py`, `tests/test_prompts.py`, `tests/test_llm.py`.
- **Delivers**: Prompt with system rules (answer only from sources, cite as [n], fixed refusal
  sentence), numbered sources with document name and page, context token budget that drops the
  lowest-ranked chunks first; Ollama client with timeout, returning answer, model, tokens used;
  connection errors, timeouts, and bad responses mapped to a "generation unavailable" error.
- **Done when**: Tests confirm prompt contents and budget trimming, and that each LLM failure mode
  maps to the right error (using a mocked HTTP transport).

### Stage 9 - Question answering endpoint

**Goal**: The full question-to-cited-answer flow.

- **Spec**: User Stories 2 and 4, FR-015, edge case 3.
- **Files**: `services/qa.py`, `api/ask.py`, `tests/test_ask.py`, `tests/test_error_handling.py`.
- **Delivers**: `POST /ask` returning answer, sources, model, tokens used; when no chunks pass the
  threshold, return "I don't have information about that" without calling the LLM.
- **Done when**: End-to-end test (upload -> process -> ask) passes with fakes; every edge case in
  the spec table has a passing test.

### Stage 10 - Containerization

**Goal**: One command brings up the whole system.

- **Spec**: SC-005.
- **Files**: `Dockerfile`, `docker-compose.yml`.
- **Delivers**: Image with CPU-only PyTorch and the embedding model pre-downloaded; compose with
  `app` and `ollama` services, persistent volumes, and automatic pull of the LLM model.
- **Done when**: `docker compose up` works from a clean checkout and the quickstart scenarios pass
  against the container.

### Stage 11 - Real-document evaluation and tuning

**Goal**: Verify quality with real documents, not just fakes.

- **Spec**: SC-001 to SC-003.
- **Delivers**: Run the quickstart with 2-3 real PDFs; calibrate the similarity threshold; iterate
  on the prompt; record observations and final values in `research.md`.
- **Done when**: Known-answer questions return correct, cited answers; off-topic questions get the
  refusal; a 10-page PDF processes in under 60 seconds.

### Stage 12 - Documentation and final review

**Goal**: Submission-ready repository.

- **Files**: `README.md`, `AI_USAGE.md`.
- **Delivers**: README (run, test, models, chunking and prompt reasoning, curl examples);
  AI_USAGE from notes collected during every stage; final walk through `spec.md` checking every
  FR, edge case, and success criterion.
- **Done when**: Every spec item is checked off and you can explain every file.

## Complexity Tracking

No constitution violations. Deliberately avoided: a separate vector DB container, a task queue
(Celery/RQ), an ORM migration tool, and LangChain/LlamaIndex - each would add complexity the spec
does not need and hide the RAG logic that is being evaluated.
