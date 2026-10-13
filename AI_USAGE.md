# AI Tool Usage

> Working draft, updated at the end of every stage (tagged by stage). Before submitting: keep the
> strongest 1-2 cases per section and rewrite them in your own words.

## Tools I Used

- Cursor with Claude (agent mode) for planning, Spec-Kit documents, and code.

## What Helped Most

- **Turning the brief into a complete spec (Stage 0).** The AI drafted user stories, functional
  requirements, and an edge-case table from the assignment, and found gaps the brief does not
  mention (asking about a document that is still processing, empty questions, `top_k` out of range).
- **Catching model-specific pitfalls early (Stage 0).** It flagged that `all-MiniLM-L6-v2` (the most
  common default suggestion) reads at most 256 tokens, so 500-token chunks would be half-ignored,
  which led to `bge-small-en-v1.5` (512 tokens). It also flagged that Ollama silently truncates
  prompts beyond its default context window, so `num_ctx` must be set explicitly.
- **Explaining framework behavior that shapes the design (Stage 1).** Starlette re-raises unexpected
  exceptions after the 500 handler runs, so the handler must not log (the server already does) and
  the test needs `raise_server_exceptions=False`.
- **Generating test data as code instead of binary files (Stage 3).** Test PDFs (multi-page, blank,
  password-protected, permission-only encrypted, zero pages) are built per test with fpdf2/pypdf,
  so every fixture is readable and reviewable.
- **Proving a hard limit with the real tokenizer (Stage 4).** Besides fast word-counter tests, one
  test chunks number- and abbreviation-heavy text plus a 2,000-word unpunctuated run with the real
  bge tokenizer and asserts every chunk fits in 512 tokens including special tokens.

- **A fake embedder that keeps tests fast and deterministic (Stage 5).** A bag-of-words vector
  hashed with `md5` makes texts that share words similar, so ranking can be tested without the real
  model; 130 tests run offline in about 8 seconds (after Stage 6). Python's built-in `hash()` would have made tests
  flaky, because it is randomized per process.
- **Turning my review concerns into measurements (Stage 5).** I asked for four checks: model
  loaded once, explicit normalization, tokenizer edge cases, and Chroma under concurrency. The AI
  answered each with a probe script (constructor count, memory before and after 100 rounds, vector
  norms, timings, a 9-thread stress run) and then locked the results in regression tests, e.g. a
  fake `sentence_transformers` module that proves one model load and `normalize_embeddings=True`.
  Two findings went into the design docs: the app must run as a single uvicorn worker (Chroma does
  not support several processes on one directory), and embedding, not chunking, dominates
  ingestion time (about 0.8 s per chunk on CPU).
- **Reading framework source before designing around it (Stage 6).** Before writing the upload
  size middleware, the AI read FastAPI's body-parsing code and found that any exception raised
  while a body is parsed becomes a generic `400 "There was an error parsing the body"`. A
  middleware that just raised an error on overflow would therefore have returned 400, not 413.
  The final middleware sends the 413 itself and discards what the app sends afterwards, and a
  test drives it with raw ASGI messages to prove exactly one response goes out.

## What I Had to Fix

- **Upload size limit enforced too late (Stage 0).** The AI's first design checked the file size
  inside the upload handler. I required the limit to be enforced at the entrance. That exposed that
  FastAPI reads the whole multipart body before the handler runs, so the final design adds a
  middleware that rejects oversized requests before reading the body, keeps the handler check as a
  second layer, and moves full text extraction to the background (the request only checks for the
  first page with text).
- **Chunks forbidden from crossing pages (Stage 0).** The AI chose "chunks never cross page
  boundaries" for exact single-page citations. I rejected it because it splits paragraphs at page
  breaks and leaves short leftover chunks. Final design: chunks cross pages, and each stores
  `page_start` and `page_end`, so citations stay precise ("pages 3-4").
- **A database constraint that silently allowed bad data (Stage 2).** The AI wrote the chunk
  page-range CHECK as `(page_start IS NULL AND page_end IS NULL) OR (page_start >= 1 AND page_end >=
  page_start)`. A test inserting `page_start = 1, page_end = NULL` showed the database accepted it:
  comparisons with NULL evaluate to *unknown*, and a CHECK only rejects rows when the result is
  *false*. Fixed by adding explicit `IS NOT NULL` checks.
- **A configuration check that was too loose (Stage 1).** The planned rule was
  `max_context_tokens < llm_num_ctx`, but the context window must also hold the answer. The
  implemented rule is `max_context_tokens + llm_num_predict < llm_num_ctx`.

## What AI Struggled With

- **Library APIs drifting between versions (Stages 1 and 5).** Answers from memory did not always
  match the installed versions: Starlette 1.8 deprecates `httpx` in its test client, Sentence
  Transformers 6.1 renamed the dimension method (`get_embedding_dimension`), and ChromaDB 1.5
  configures cosine distance through `configuration=` and rejects collection names shorter than 3
  characters. The fix was to probe the installed library with a small script before designing
  against it, and to pin exact versions.
- **Tests that pass on the first run prove nothing by themselves (Stages 3-4).** All chunking tests
  passed immediately, so the code was deliberately broken (overlap disabled, page-break joining
  disabled) to confirm the tests fail: 5 and 1 tests failed respectively. The same was done in
  Stage 5: with `normalize_embeddings=False` the new embedder test fails.
- **A race condition in its own earlier design (Stages 0 and 6).** The Stage 0 design for
  "delete while processing" relied on the chunk insert failing on the foreign key, then cleaning
  up. Designing Stage 6 in detail showed a gap: a batch written between the delete's two steps
  leaves orphan vectors. The fix is one lock shared by batch writes and deletes, with the slow
  embedding work outside it. A two-thread test races a delete against a batch write; with the lock
  removed it fails with 5 orphan vectors.
- **Tests that could not fail (Stage 6).** The first delete-during-processing test was
  single-threaded, so it passed with or without the lock; the two-thread test above was added to
  prove the lock is needed. Removing the "delete the saved file on rejection" line made 4 tests fail, but the
  page-limit and size-limit tests still passed because they did not check the uploads folder; they
  now do.
- **A benchmark that measured the wrong thing (Stage 5).** The first probe reported the same token
  count for "1 MB" and "10 MB" texts: it sliced a 1.1 MB string to 10 MB, which changes nothing.
  The suspicious number exposed it; the corrected run showed counting scales linearly (about 1.7 s
  per MB). Benchmark output needs a sanity check before it is believed.
- **Windows shell quirks.** Inline Python passed through PowerShell lost its `$` characters
  (`$in` in a Chroma filter), so probes were moved into script files.
