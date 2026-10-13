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
  disabled) to confirm the tests fail: 5 and 1 tests failed respectively.
- **Windows shell quirks.** Inline Python passed through PowerShell lost its `$` characters
  (`$in` in a Chroma filter), so probes were moved into script files.
