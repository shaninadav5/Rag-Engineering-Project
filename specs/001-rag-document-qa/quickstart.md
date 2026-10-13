# Quickstart: Manual Validation Scenarios

**Input**: [spec.md](./spec.md), [contracts/openapi.yaml](./contracts/openapi.yaml)

These scenarios are run by hand against a running server (Stages 10 and 11). Automated tests cover
the same behavior with fakes; this checks the real models.

Examples use `curl.exe` so they work in PowerShell (where plain `curl` is an alias for
`Invoke-WebRequest`) as well as in Git Bash.

## Setup

```powershell
docker compose up --build        # or locally: uvicorn app.main:app --reload
$BASE = "http://localhost:8000"
```

Prepare `tests/fixtures/` style samples, or use any real PDF. Below, `report.pdf` contains a known
fact, e.g. "Q3 revenue grew 12%".

## Scenario 1 - Upload and process (User Story 1)

```powershell
curl.exe -s -X POST "$BASE/documents" -F "file=@report.pdf" -F "name=Q3 Report"
```

Expect `202`, `status: "pending"`, `name: "Q3 Report"`, `pages` set. Copy the `id`.

```powershell
curl.exe -s "$BASE/documents/<id>"
```

Repeat until `status` is `completed`. Expect `chunks > 0` and `processed_at` set.

## Scenario 2 - Ask a question (User Story 2)

Write the body to a file to avoid PowerShell quoting issues:

```powershell
'{"question": "How much did revenue grow in Q3?"}' | Out-File -Encoding ascii ask.json
curl.exe -s -X POST "$BASE/ask" -H "Content-Type: application/json" --data "@ask.json"
```

Expect the answer to state 12% with a citation like `[1]`; `sources[0]` is from "Q3 Report" with
`page` / `page_end` set and a `relevance_score`; `model` and `tokens_used > 0` present.

## Scenario 3 - Off-topic question (edge case 3)

Body: `{"question": "What is the capital of Australia?"}`

Expect `answer: "I don't have information about that."`, `sources: []`, `tokens_used: 0`.

## Scenario 4 - Scoped question (User Story 4)

Upload a second document, then:

Body: `{"question": "...", "document_ids": ["<id of Q3 Report>"], "top_k": 3}`

Expect at most 3 sources, all with the Q3 Report `document_id`.

## Scenario 5 - List and delete (User Story 3)

```powershell
curl.exe -s "$BASE/documents"
curl.exe -s -o NUL -w "%{http_code}" -X DELETE "$BASE/documents/<id>"
```

Expect the list to show both documents, then `204`. Re-ask Scenario 2's question: no source from
the deleted document. `GET /documents/<id>` now returns `404 DOCUMENT_NOT_FOUND`.

## Scenario 6 - Rejections (edge cases 1, 2, 5)

| Action | Expected |
|--------|----------|
| Upload an empty `empty.txt` | `400 EMPTY_DOCUMENT` |
| Upload a scanned (image-only) PDF | `400 EMPTY_DOCUMENT` |
| Upload `notes.docx` | `415 UNSUPPORTED_FILE_TYPE` listing `.pdf, .txt, .md` |
| Upload a file above `RAG_MAX_UPLOAD_MB` | Rejected immediately, not after the full upload. Expect `413 FILE_TOO_LARGE`; some clients (including curl) may instead report "connection reset" or "broken pipe", because the server answers and closes before reading the body. Either outcome is correct; the automated test asserts the `413` |
| Upload a PDF with more pages than `RAG_MAX_PAGES` | `413 TOO_MANY_PAGES` |
| While a large valid PDF uploads, call `GET /health` | Responds immediately (validation does not block the server) |

## Scenario 7 - Bad document ids (edge case 4)

| Action | Expected |
|--------|----------|
| Ask with `document_ids: ["does-not-exist"]` | `404 DOCUMENT_NOT_FOUND` with `missing_ids` |
| Ask with the id of a document still `processing` | `409 DOCUMENT_NOT_READY` |

## Scenario 8 - LLM down (edge case 6)

```powershell
docker compose stop ollama
```

Ask any on-topic question. Expect `503 LLM_UNAVAILABLE`; `GET /health` still returns `200`.
Restart with `docker compose start ollama`.

## Scenario 9 - Validation

| Body | Expected |
|------|----------|
| `{"question": "   "}` | `422 VALIDATION_ERROR` |
| `{"question": "x", "top_k": 0}` | `422 VALIDATION_ERROR` |
| `{"question": "x", "document_ids": []}` | `422 VALIDATION_ERROR` |

## Scenario 10 - Performance (SC-001)

Upload a 10-page text PDF and time from upload to `completed`. Expect under 60 seconds on CPU.
