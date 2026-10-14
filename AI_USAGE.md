# AI Tool Usage

## Tools I Used

- Cursor with Claude (agent mode) for planning, Spec-Kit documents, and code.

## What Helped Most

- **Early identification of token limitations in embedding models (Stage 0).** The assistant flagged that the popular default, `all-MiniLM-L6-v2`, reads at most 256 tokens, so a 500-token chunk would have been silently cut in half. We switched to `bge-small-en-v1.5` (512 tokens) and always send `num_ctx` to Ollama, because Ollama otherwise drops the start of a prompt that exceeds its default window.
- **Creating a mocked embedding model for fast offline testing (Stage 5).** A bag-of-words vector hashed with `md5` (not Python's per-process `hash()`) makes texts that share words similar. About 200 tests run offline, without the real model, in about 15 seconds. The tests that load the real embedding model are separate and slower.

## What I Had to Fix

- **Moving file size enforcement to an entrance middleware (Stage 0).** The first design checked the file size inside the upload endpoint. I required the limit at the entrance. That showed FastAPI writes the whole multipart body to a temporary file before the endpoint runs. An ASGI middleware in the app rejects a request whose `Content-Length` is already over the limit without reading the body. A chunked upload is read only until it crosses the limit, then cut off. The endpoint still checks the saved file against 20 MB.
- **Fixing container security and running as a non-root user (Stage 10).** The first Dockerfile ran the server as root. The process that serves requests now runs as `appuser` (uid 1000). The entrypoint stays root only long enough to `chown` the `./data` bind mount, then drops privileges with `gosu`. Without that `chown`, a root-owned mount would not be writable by `appuser`.

## What AI Struggled With

- **Interface drift and library version changes (Stages 1, 5, and 7).** Answers from memory did not match the installed libraries: Sentence Transformers 6.1 renamed the dimension method, ChromaDB 1.5 rejects collection names shorter than 3 characters and configures cosine distance differently, and SQLAlchemy 2.1 deprecates `Result.tuples()`. The fix was to probe the installed library with a small script before writing against it, and to pin exact versions.
- **Preventing race conditions during concurrent document deletions (Stages 0 and 6).** A delete that runs while a batch of chunks is being stored can leave orphan vectors in Chroma. One lock covers the batch write and the delete. Embedding stays outside the lock, so a slow batch does not block every other request for the whole embedding time.
