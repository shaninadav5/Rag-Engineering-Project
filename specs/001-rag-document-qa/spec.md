# Feature Specification: RAG Document Q&A

**Feature Branch**: `001-rag-document-qa`
**Created**: 2026-10-13
**Status**: Approved (2026-10-13)
**Input**: Technical assessment brief `junior-mid-rag-qa.md` - a system where users upload documents
and ask questions about them; answers are generated from retrieved passages with source citations.

## User Scenarios & Testing

### User Story 1 - Upload and process a document (Priority: P1)

A user uploads a PDF, TXT, or Markdown file, optionally giving it a display name. The system
accepts it, processes it in the background (extract text, split into overlapping chunks, embed,
store), and the user can see when processing has finished.

**Why this priority**: Nothing can be asked until documents are in the system.

**Independent Test**: Upload a sample file, poll its metadata until the status is `completed`,
and confirm the chunk count is greater than zero.

**Acceptance Scenarios**:

1. **Given** a valid PDF, **When** the user uploads it with name "Q3 Report", **Then** the system
   returns the new document with status `pending` and display name "Q3 Report".
2. **Given** a valid TXT file uploaded without a name, **When** processing completes, **Then** the
   display name equals the original filename and the status is `completed`.
3. **Given** a multi-page PDF, **When** processing completes, **Then** each chunk records the
   range of pages it spans (start and end page).
4. **Given** a paragraph that continues from one PDF page onto the next, **When** processing
   completes, **Then** that paragraph is not split at the page break (as long as it fits in a chunk).

---

### User Story 2 - Ask a question and get a cited answer (Priority: P1)

A user asks a natural-language question. The system finds the most relevant passages across the
uploaded documents and returns an answer based only on those passages, with citations showing
which document, page, and passage each part of the answer came from, plus a relevance score.

**Why this priority**: This is the core value of the product.

**Independent Test**: With one processed document containing a known fact, ask about that fact and
confirm the answer contains it and the sources include the passage holding it.

**Acceptance Scenarios**:

1. **Given** a processed document stating "Q3 revenue grew 12%", **When** the user asks "How much
   did revenue grow in Q3?", **Then** the answer states 12% and cites that document.
2. **Given** processed documents, **When** the user asks about a topic none of them cover,
   **Then** the answer is "I don't have information about that" and no sources are returned.
3. **Given** any answered question, **Then** the response includes the model name and the number
   of tokens used.

---

### User Story 3 - Manage documents (Priority: P2)

A user lists all uploaded documents, views a single document's metadata (pages, chunks, upload
date, status), and deletes documents they no longer need. Deleting a document removes it and all
its searchable content.

**Why this priority**: Needed to keep the knowledge base accurate, but not to demonstrate Q&A.

**Independent Test**: Upload two documents, list them, delete one, confirm it no longer appears in
the list and its content is never returned as a source.

**Acceptance Scenarios**:

1. **Given** two uploaded documents, **When** the user lists documents, **Then** both appear with
   id, name, filename, chunk count, upload date, and status.
2. **Given** a processed document, **When** the user deletes it and asks a question it previously
   answered, **Then** none of its passages appear in the sources.

---

### User Story 4 - Scope a question (Priority: P2)

A user restricts a question to specific documents and/or chooses how many passages to retrieve.

**Why this priority**: Improves precision when many documents are loaded.

**Independent Test**: Upload two documents with different facts, ask with a filter on one, confirm
all sources come from that document.

**Acceptance Scenarios**:

1. **Given** documents A and B, **When** the user asks with `document_ids=[A]`, **Then** every
   source belongs to A.
2. **Given** `top_k=3`, **When** the user asks, **Then** at most 3 sources are returned.

---

### Edge Cases

| # | Situation | Expected behavior | Met |
|---|-----------|-------------------|-----|
| 1 | Empty document (no bytes, or no extractable text) | Rejected with a clear error saying the document is empty | [x] |
| 2 | Document larger than the configured size limit, or with more pages than the page limit | Rejected at the entrance, before the upload is fully received where possible, with an error stating the limit | [x] |
| 3 | No passage is relevant enough to the question | Answer "I don't have information about that", no sources, LLM not called | [x] |
| 4 | Question filtered to a document id that does not exist | Clear error naming the unknown id(s) | [x] |
| 5 | Unsupported file type | Rejected with an error listing the supported types (PDF, TXT, MD) | [x] |
| 6 | LLM unavailable, timing out, or erroring | Clear "generation service unavailable" error; service keeps running | [x] |
| 7 | Embedding fails during processing | Document marked `failed` with the reason; user may delete and re-upload. If the question itself cannot be embedded, a clear "embedding unavailable" error | [x] |

Additional edge cases:

- [x] Question filtered to a document that exists but has not finished processing: clear error stating
  the document is not ready.
- [x] Question asked when no documents have been processed: same as edge case 3.
- [x] Empty or whitespace-only question: rejected with a validation error.
- [x] `top_k` outside the allowed range: rejected with a validation error.
- [x] Deleting a document that does not exist: clear "not found" error.

## Requirements

### Functional Requirements

**Document management**

- [x] **FR-001**: System MUST accept uploads of PDF, TXT, and Markdown files, with an optional display
  name that defaults to the original filename.
- [x] **FR-002**: System MUST list all documents with id, display name, filename, chunk count, upload
  date, and processing status.
- [x] **FR-003**: System MUST return a single document's metadata: id, display name, filename, file
  type, page count (when applicable), chunk count, upload date, status, and failure reason (if any).
- [x] **FR-004**: System MUST delete a document together with all of its chunks, embeddings, and stored
  original file, including while the document is still processing.
- [x] **FR-004a**: System MUST keep the original uploaded file on disk for the document's lifetime.

**Document processing**

- [x] **FR-004b**: System MUST enforce the upload size limit before the request body is fully read
  whenever the declared size exceeds it, and MUST never accept a file above the limit.
- [x] **FR-004c**: The upload request MUST perform only lightweight validation (type, size, page limit,
  readability, presence of text) and MUST NOT block other requests while doing so.
- [x] **FR-005**: System MUST extract text from uploaded documents, preserving page numbers for PDFs.
- [x] **FR-006**: System MUST split text into chunks of a configurable size (default ~500 tokens) with
  configurable overlap (default ~50 tokens), preserving paragraph boundaries where possible.
  Chunks MAY cross page boundaries so that text continuing across a page break stays together.
- [x] **FR-007**: Each chunk MUST record its document, its position (chunk index), the start and end
  page it spans (when applicable), its token count, and a reference to its embedding.
- [x] **FR-008**: System MUST generate an embedding for every chunk and store it for similarity search.
- [x] **FR-009**: Processing MUST run without blocking the upload response and MUST track status:
  `pending` → `processing` → `completed` or `failed`.

**Question answering**

- [x] **FR-010**: System MUST retrieve the top-k chunks most similar to the question (k configurable,
  default 5, overridable per request).
- [x] **FR-011**: System MUST support restricting retrieval to a list of document ids. Without a
  filter, only `completed` documents are searched.
- [x] **FR-012**: System MUST discard chunks below a configurable similarity threshold.
- [x] **FR-013**: System MUST generate an answer using only the retrieved chunks, instructing the model
  to cite sources.
- [x] **FR-014**: System MUST keep the prompt within the model's context limit by capping the amount of
  retrieved text included.
- [x] **FR-015**: Each response MUST include the answer, the sources (document id, document name,
  chunk text, start page, end page, relevance score), the model name, and tokens used.

**Errors**

- [x] **FR-016**: System MUST handle all edge cases listed above with clear, specific error messages.

### Key Entities

- **Document**: An uploaded file. Attributes: unique id, original filename, display name, file type,
  page count, total chunks, upload timestamp, processing status, failure reason.
- **Chunk**: A passage of a document used for retrieval. Attributes: unique id, owning document,
  chunk index, text, start page, end page, token count, embedding reference.

## Success Criteria

### Measurable Outcomes

- [x] **SC-001**: A 10-page text PDF finishes processing in under 60 seconds on a typical laptop CPU.
- [x] **SC-002**: For a question whose answer is in a single known passage, that passage is among the
  returned sources.
- [x] **SC-003**: For questions unrelated to any uploaded document, the system answers "I don't have
  information about that" rather than inventing an answer.
- [x] **SC-004**: Every edge case in the table above produces its expected behavior, verified by
  automated tests.
- [x] **SC-005**: The full system starts with a single `docker compose up` and requires no API keys.

## Assumptions

- Single user, no authentication or multi-tenancy.
- Documents are primarily English.
- Scanned (image-only) PDFs are out of scope; no OCR. They are treated as empty documents.
- Maximum upload size defaults to 20 MB and maximum PDF page count to 1000 (both configurable).
- Uploading the same file twice creates two separate documents.
- Conversation history, streaming, reranking, and hybrid search are out of scope for this version.

## Clarifications

### Session 2026-10-13

- Q: Can a document be deleted while it is still processing? → A: Yes. Processing is abandoned and
  no chunks, embeddings, or files are left behind.
- Q: Without a `document_ids` filter, are documents still processing searched? → A: No. Only
  `completed` documents are searched; others are skipped silently.
- Q: Is the original uploaded file kept after processing? → A: Yes. It is stored on disk and
  deleted together with the document.
