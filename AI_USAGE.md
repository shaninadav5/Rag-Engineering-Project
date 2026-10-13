# AI Tool Usage

> Working draft, updated at the end of every stage. Rewrite in your own words before submitting.

## Tools I Used

- Cursor with Claude (agent mode) for planning, Spec-Kit documents, and code.

## What Helped Most

- **Turning the brief into a complete spec.** The AI drafted user stories, functional requirements,
  and an edge-case table from the assignment, and found gaps the brief does not mention (asking
  about a document that is still processing, empty questions, `top_k` out of range).
- **Catching model-specific pitfalls early.** It flagged that `all-MiniLM-L6-v2` (the most common
  default suggestion) reads at most 256 tokens, so 500-token chunks would be half-ignored, which led
  to `bge-small-en-v1.5` (512 tokens). It also flagged that Ollama silently truncates prompts beyond
  its default context window, so `num_ctx` must be set explicitly.

## What I Had to Fix

- **Upload size limit enforced too late.** The AI's first design checked the file size inside the
  upload handler. I required the limit to be enforced at the entrance. That exposed that FastAPI
  reads the whole multipart body before the handler runs, so the final design adds a middleware
  that rejects oversized requests before reading the body, keeps the handler check as a second
  layer, and moves full text extraction to the background (the request only checks for the first
  page with text).
- **Chunks forbidden from crossing pages.** The AI chose "chunks never cross page boundaries" for
  exact single-page citations. I rejected it because it splits paragraphs at page breaks and leaves
  short leftover chunks. Final design: chunks cross pages, and each stores `page_start` and
  `page_end`, so citations stay precise ("pages 3-4").

- **A database constraint that silently allowed bad data.** The AI wrote the chunk page-range
  CHECK as `(page_start IS NULL AND page_end IS NULL) OR (page_start >= 1 AND page_end >= page_start)`.
  A test inserting `page_start = 1, page_end = NULL` showed the database accepted it: comparisons
  with NULL evaluate to *unknown*, and a CHECK only rejects rows when the result is *false*. Fixed
  by adding explicit `IS NOT NULL` checks. Lesson: write the failing case as a test, don't trust
  the constraint by reading it.

## What AI Struggled With

- *(to fill in during implementation)*
