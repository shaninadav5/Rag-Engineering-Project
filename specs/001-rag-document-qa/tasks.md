# Tasks: RAG Document Q&A

**Input**: [plan.md](./plan.md), [spec.md](./spec.md)

Check a task off only when its code and tests are done and reviewed. Each stage ends with a commit.

## Stage 0 - Design artifacts

- [x] T001 Write `research.md`: chunking, embedding model, vector store, retrieval, prompt design
- [x] T002 Write `data-model.md`: Document and Chunk fields, types, constraints, status transitions
- [x] T003 Write `contracts/openapi.yaml`: all endpoints, schemas, error responses and status codes
- [x] T004 Write `quickstart.md`: curl scenarios covering every user story and edge case
- [x] T005 Review and approve the design artifacts
- [x] T005a Start `AI_USAGE.md` and record AI corrections at the end of every stage

## Stage 1 - Foundation

- [x] T006 Create scaffold: `pyproject.toml`, requirements files, ignore files, `.env.example`
- [x] T007 `app/config.py` settings with chunk-overlap validation
- [x] T008 `app/main.py` app factory with `/health`
- [x] T009 Create virtual environment and install `requirements-dev.txt`; pin direct dependencies
- [x] T009a Add settings from research.md: `max_pages`, `top_k_max`, `max_question_chars`,
  `llm_temperature`, `llm_num_ctx`, `llm_num_predict`
- [x] T010 `app/errors.py` domain exceptions and HTTP exception handlers using the
  `{"error": {code, message, details}}` envelope, including request-validation errors
- [x] T011 `app/db/database.py` engine, session factory, `Base`, table creation on startup
- [x] T012 `tests/conftest.py` with temporary data directory and `TestClient`
- [x] T013 Run tests; verify `/health` and `/docs` with Uvicorn; commit

## Stage 2 - Data model

- [x] T014 `app/db/models.py` `DocumentStatus`, `Document`, `Chunk` with cascade delete
- [x] T015 `app/schemas.py` document response schemas
- [x] T016 Tests: create, read, delete document with chunks; commit

## Stage 3 - Text extraction

- [x] T017 Add fixtures: multi-page PDF, paragraph crossing a page break, image-only PDF, corrupt PDF,
  TXT, MD, non-UTF-8 TXT, empty file (generated per test by `tests/pdf_factory.py` with fpdf2)
- [x] T018 `app/services/extraction.py` full per-page `extract` for PDF, TXT, MD
- [x] T019 Lightweight `validate`: readability, page count vs `max_pages`, stop at first page with text
- [x] T019a Errors: unsupported type, unreadable, too many pages, empty document
- [x] T020 `tests/test_extraction.py` (including early exit of `validate`); commit

## Stage 4 - Chunking

- [x] T021 `app/services/chunking.py` paragraph splitting and token counting with the model tokenizer
- [x] T022 Packing units across the whole document (page breaks are not boundaries) with overlap
- [x] T023 Sentence-level splitting for oversized paragraphs; page tags -> `page_start` / `page_end`
- [x] T024 `tests/test_chunking.py` (size limit, overlap, paragraph preservation, cross-page chunk and
  page range, empty); commit

## Stage 5 - Embeddings and vector store

- [ ] T025 `app/services/embeddings.py` `Embedder` interface + Sentence Transformers implementation
- [ ] T025a Load the model at startup; fail fast if it cannot load, if `embedding_dim` does not match,
  or if `chunk_size` exceeds `max_seq_length - 2`
- [ ] T026 `tests/fakes.py` deterministic fake embedder
- [ ] T027 `app/services/vector_store.py` add, search with filter, delete by document
- [ ] T028 `tests/test_vector_store.py` (ranking, top-k, filter, delete)
- [ ] T029 Manual check with the real embedding model; commit

## Stage 6 - Ingestion pipeline and document API

- [ ] T030 `app/dependencies.py` providers for session, embedder, vector store, LLM
- [ ] T031 `app/services/ingestion.py` background pipeline with status updates and failure reason
- [ ] T031a Upload-size middleware: reject on `Content-Length` over the limit before reading the body;
  count streamed bytes when the header is missing or wrong
- [ ] T032 `app/api/documents.py` upload (`202`): exact size check, lightweight `validate` in a worker
  thread, file storage, schedule background extract + chunk + embed
- [ ] T032a On startup, mark documents left in `pending`/`processing` as `failed` ("interrupted by restart")
- [ ] T033 List, get metadata, delete (SQLite + Chroma + file, safe mid-processing; the background
  task treats "document deleted" as a clean stop, not a failure)
- [ ] T034 `tests/test_documents_api.py` (happy paths, rejections incl. oversized body rejected by
  middleware, too many pages, embedding failure); commit

## Stage 7 - Retrieval

- [ ] T035 `app/services/retrieval.py` query embedding, top-k, threshold, enrichment
- [ ] T036 `document_ids` validation: unknown and not-completed documents
- [ ] T037 `tests/test_retrieval.py` (known fact ranks first, filters, invalid ids); commit

## Stage 8 - Prompt and LLM client

- [ ] T038 `app/services/prompts.py` system rules, numbered sources, context budget
- [ ] T039 `app/services/llm.py` Ollama client with timeout and error mapping
- [ ] T040 `tests/test_prompts.py` and `tests/test_llm.py` (mocked HTTP transport); commit

## Stage 9 - Question answering endpoint

- [ ] T041 `app/services/qa.py` retrieve -> prompt -> generate -> response; no-information path
- [ ] T042 `app/api/ask.py` `POST /ask` with request validation
- [ ] T043 `tests/test_ask.py` end-to-end upload -> process -> ask
- [ ] T044 `tests/test_error_handling.py` covering every edge case in the spec; commit

## Stage 10 - Containerization

- [ ] T045 `Dockerfile` with CPU-only PyTorch and pre-downloaded embedding model
- [ ] T046 `docker-compose.yml` with `app`, `ollama`, volumes, model pull
- [ ] T047 Run quickstart against containers; commit

## Stage 11 - Real-document evaluation and tuning

- [ ] T048 Process 2-3 real PDFs; measure processing time
- [ ] T049 Calibrate similarity threshold; iterate on the prompt
- [ ] T050 Record findings and final values in `research.md`; commit

## Stage 12 - Documentation and final review

- [ ] T051 `README.md`
- [ ] T052 `AI_USAGE.md` from notes collected during each stage
- [ ] T053 Walk through `spec.md` and check off every FR, edge case, and success criterion; commit
