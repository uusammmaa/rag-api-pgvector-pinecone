# RAG Document API

[![CI](https://github.com/uusammmaa/rag-api-pgvector-pinecone/actions/workflows/ci.yml/badge.svg)](https://github.com/uusammmaa/rag-api-pgvector-pinecone/actions/workflows/ci.yml)

**Ask questions about a PDF and get answers grounded in its text, with the retrieval quality measured instead of assumed.**

A FastAPI service that ingests PDFs (extract, chunk, embed), stores them in Pinecone or Postgres/pgvector, and streams Claude answers built only from the retrieved excerpts. It ships with an eval harness that scores retrieval (precision@k, recall@k, MRR) and answers (faithfulness, relevance via LLM-as-judge) across chunking strategies and k values, and runs in CI without API keys.

## Architecture

```mermaid
flowchart LR
    subgraph ingest["Ingest: POST /upload"]
        A[PDF upload] --> B["Validate<br/>size, type, %PDF magic"]
        B --> C["Extract text<br/>pypdf + page map"]
        C --> D["Chunk<br/>fixed | recursive | heading"]
        D --> E["Embed<br/>OpenAI or offline hash"]
    end

    subgraph query["Answer: POST /chat, POST /query"]
        Q[Question + doc_id] --> QE[Embed question]
        QE --> R["Top-k search<br/>scoped to doc_id"]
        R --> P["Grounded prompt<br/>numbered excerpts"]
        P --> L["Claude<br/>claude-haiku-5-5"]
        L --> O["text/plain stream<br/>or JSON + sources"]
    end

    E --> VS{{VectorStore interface}}
    R --> VS
    VS --- PC[(Pinecone)]
    VS --- PG[("Postgres + pgvector<br/>local or Supabase")]
    VS --- MEM[(In-memory)]

    subgraph evals["Evals: python -m evals.run"]
        DS["38 labelled questions<br/>3 public-domain PDFs"] --> H["Same extract, chunk,<br/>embed and search code"]
        H --> M["precision@k, recall@k, MRR"]
        H --> J["Judge: faithfulness,<br/>relevance (claude-opus-5-5)"]
        M & J --> RES[evals/RESULTS.md]
    end
```

Cross-cutting: every request gets an `X-Request-ID` (propagated if the caller sends one) that appears on the response and on every log line; errors share one JSON envelope; providers are built lazily so a missing key becomes a `503` on the request that needs it instead of a crash at startup.

## Quickstart

### Docker (no API keys needed)

```bash
docker compose up --build
```

This starts the API on `http://localhost:8000` and Postgres 17 with pgvector. With no `.env` it runs fully offline: a deterministic hash embedder and an extractive "answerer" that quotes the best matching retrieved sentences. Good for trying the API and the pgvector path; not a substitute for real models.

```bash
curl -F file=@evals/corpus/us_constitution.pdf localhost:8000/upload
# {"doc_id":"5b1c...","filename":"us_constitution.pdf","chunks":36,"pages":8,"vector_store":"pgvector","chunk_strategy":"recursive"}

curl -N localhost:8000/chat -H 'content-type: application/json' \
  -d '{"question": "Who has the power to declare war?", "doc_id": "5b1c..."}'
```

To use real models in the compose stack, copy `.env.example` to `.env`, set `EMBEDDING_PROVIDER=openai`, `LLM_PROVIDER=anthropic` and the two API keys.

### Local development

Requires Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
make install            # runtime + dev dependencies
cp .env.example .env    # fill in keys for the providers you select
make run                # uvicorn app.main:app --reload
```

OpenAPI docs: `http://localhost:8000/docs`.

| Command | What it does |
|---|---|
| `make test` | Unit and API tests, offline |
| `make test-integration` | Adds the pgvector round trip against `TEST_DATABASE_URL` (run `make up` first) |
| `make lint` / `make format` | Ruff |
| `make eval` | Mock evals, writes `evals/RESULTS.md` |
| `make eval-live` | Evals with OpenAI embeddings, Claude answers and the Claude judge |
| `make up` / `make down` | Docker compose stack |

## Switching between Pinecone and pgvector

The pipeline only talks to the `VectorStore` interface (`app/vectorstores/base.py`). Pick the default with `VECTOR_STORE`, or override it per request: `POST /upload` takes a `vector_store` form field and `POST /chat` / `POST /query` take a `vector_store` JSON field. Retrieval always filters by `doc_id`, so each question only sees chunks from its own upload.

| `VECTOR_STORE` | Needs | Notes |
|---|---|---|
| `pinecone` | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Create a **cosine** index with dimension `EMBEDDING_DIM` (1536 for `text-embedding-3-small`). Chunk text and metadata live in the vector metadata. |
| `pgvector` | `DATABASE_URL` | Any Postgres with the `vector` extension. The table is created on first use (`PGVECTOR_AUTO_MIGRATE=true`), or apply `db/schema.sql` yourself. |
| `supabase` | `DATABASE_URL` | Alias for `pgvector`. Use the connection string from Project Settings > Database (the pooler URI). |
| `memory` | nothing | Per-process, lost on restart. Used by tests and evals. |

Local pgvector: `docker compose up db`, then `DATABASE_URL=postgresql://rag:rag@localhost:5432/rag`.

A document lives in the store it was uploaded to: ask about it with the same `vector_store` you uploaded with.

## API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | `{"status": "ok", "version": "1.0.0"}` |
| `POST /upload` | multipart: `file` (PDF), optional `vector_store` | `doc_id`, `filename`, `chunks`, `pages`, `vector_store`, `chunk_strategy` |
| `POST /chat` | `{"question", "doc_id", "vector_store"?, "top_k"?}` | Streamed `text/plain` answer; `X-Retrieved-Chunks` header |
| `POST /query` | same as `/chat` | `{"answer", "sources": [{chunk_id, text, score, page, section}], "model"}` |

Errors always look like this, with the same `request_id` that is in the `X-Request-ID` header and the logs:

```json
{"error": {"code": "payload_too_large", "message": "File exceeds the 20 MB limit.", "request_id": "9f2c..."}}
```

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Empty file, question over `MAX_QUESTION_CHARS` |
| 404 | `document_not_found` | No chunks for that `doc_id` in that store |
| 413 | `payload_too_large` | File over `MAX_UPLOAD_MB` |
| 415 | `unsupported_media_type` | Not `.pdf`, disallowed content type, or missing `%PDF` header |
| 422 | `unprocessable_document` / `validation_error` | Unparseable or image-only PDF / invalid request fields |
| 502 | `upstream_error` | Embedding, LLM or vector store call failed (details are logged, not returned) |
| 503 | `provider_not_configured` | The selected provider has no credentials |

`/chat` embeds and retrieves before the response starts, so all of the above arrive as real status codes. If the model fails after streaming has begun, the body ends with `[error] ... (request_id=...)`.

## Configuration

All settings come from environment variables or `.env` (see `.env.example`, which holds placeholders only).

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `hash` (offline) |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `text-embedding-3-small` / `1536` | Must match the Pinecone index or pgvector column |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `mock` (offline extractive) |
| `CHAT_MODEL` / `CHAT_EFFORT` | `claude-haiku-5-5` / `low` | Answering model |
| `JUDGE_MODEL` | `claude-opus-5-5` | Eval judge |
| `VECTOR_STORE` | `pinecone` | `pinecone`, `pgvector` (`supabase`), `memory` |
| `CHUNK_STRATEGY` / `CHUNK_SIZE` / `CHUNK_OVERLAP` | `recursive` / `800` / `100` | Characters |
| `TOP_K` | `5` | Chunks per answer (requests may override, max 20) |
| `MAX_UPLOAD_MB` / `MAX_QUESTION_CHARS` | `20` / `2000` | Input limits |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma separated |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `false` | JSON logs are on in the Docker image |

## Evals

```bash
make eval                         # or: python -m evals.run
make eval-live                    # or: python -m evals.run --mode live
python -m evals.run --strategies recursive heading --k 3 5 10 --chunk-size 1200
```

**Dataset** (`evals/dataset.jsonl`): 38 questions over three public-domain PDFs chosen for different structure: the US Constitution (articles and sections), *The Elements of Style* (chapters and numbered rules) and the Declaration of Independence (unbroken prose). Each question has a reference answer and one or more verbatim evidence passages (50 in total); 11 questions need two or three passages, so recall@k is meaningful. Provenance is in `evals/corpus/README.md`.

**How it scores.** Evidence passages are anchored to character offsets in the extracted PDF text, and every chunker keeps offsets, so one set of labels scores every strategy. A chunk is relevant when it overlaps an evidence span by at least half of the shorter of the two; a span is found when the top-k chunks cover at least half of it. Answers are graded 1-5 for **faithfulness** (claims supported by the retrieved context) and **relevance** (answers the question, agrees with the reference) by `claude-opus-5-5` using a fixed rubric and structured output (`evals/judge.py`).

**Mock mode** swaps in offline components: a hash embedder (lexical, so it behaves like a weak BM25), an extractive answerer and a lexical judge. That keeps CI free and deterministic: the workflow runs it on every push, fails if the best recall@k drops below 0.7, and fails if the committed `RESULTS.md` is stale. Mock faithfulness and relevance are token-overlap proxies, and the extractive answerer is faithful by construction, so read only the retrieval columns as signal.

### Results (mock mode, chunk size 800, overlap 100)

From [`evals/RESULTS.md`](evals/RESULTS.md):

| Strategy | k | Chunks | Avg chars | Precision@k | Recall@k | MRR | Faithfulness (1-5) | Relevance (1-5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 3 | 151 | 793 | 0.237 | 0.671 | 0.561 | 5.00 | 2.24 |
| fixed | 5 | 151 | 793 | 0.163 | 0.759 | 0.585 | 5.00 | 2.31 |
| recursive | 3 | 148 | 751 | 0.237 | 0.627 | 0.544 | 5.00 | 2.26 |
| **recursive** | **5** | 148 | 751 | 0.179 | **0.798** | 0.577 | 5.00 | 2.41 |
| heading | 3 | 168 | 655 | 0.193 | 0.561 | 0.496 | 5.00 | 2.30 |
| heading | 5 | 168 | 655 | 0.147 | 0.719 | 0.531 | 5.00 | 2.48 |

What the mock run says:

- **k=5 beats k=3 for every strategy** (recall +0.09 to +0.17) at a lower precision. With a grounded prompt, missing evidence costs more than two extra excerpts, so `TOP_K=5` stays the default.
- **Recursive at k=5 has the best recall**, mostly from the Constitution (0.865 vs 0.677 fixed and 0.583 heading), where splitting on line boundaries keeps clauses intact.
- **Heading-aware loses under a lexical embedder.** It makes more, smaller chunks, and the section path it prepends ("Article 1 > Section 8") adds the same tokens to many chunks, which blurs a bag-of-words vector. Removing the prefix recovers part of the gap. Section context usually helps semantic embeddings, so this is the comparison most likely to change in a live run.
- **Chunk size matters more than strategy here.** A side sweep at 400 characters (the previous default) gave a best recall@5 of 0.671; 1200 gave 0.825 with fixed chunks, at 50% more context tokens per answer.

Live numbers need API keys and are not committed yet. `make eval-live` writes them to the same files, with embeddings cached under `evals/.cache/` so reruns only pay for generation and judging (a full run is 228 answers plus 228 judge grades).

## Design decisions and tradeoffs

- **Evidence spans, not chunk IDs, as labels.** Chunk-ID labels only score the strategy they were written for. Offset-anchored passages score any chunker and survive re-chunking, at the cost of a fuzzy "relevant" threshold (50% overlap of the shorter span), which is stated next to the results.
- **Offline providers as first-class implementations.** The hash embedder and extractive answerer sit behind the same interfaces as OpenAI and Claude, so tests, CI evals and the Docker quickstart need no keys. The cost is that mock answer scores are proxies; the report says so and live mode is one flag away.
- **Retrieve before streaming.** `/chat` finishes embedding and search before the first byte, so validation, unknown documents and provider outages return proper status codes. Only generation streams, and a mid-stream failure ends the body with an `[error]` line because the status line is already sent. Server-sent events would carry that more cleanly; plain text keeps existing clients working.
- **pgvector over Postgres, not the Supabase REST client.** One code path covers local Postgres, CI and Supabase; writes are transactional upserts keyed by chunk ID, so retried writes cannot duplicate rows. It needs a direct connection string and a connection pool (`psycopg_pool`), and on Supabase you should use the pooler URI.
- **Exact search per document, no HNSW index.** Every query filters on `doc_id`. An approximate index applies that filter after picking candidates and can return fewer than k rows for a small document in a large table. The B-tree on `doc_id` plus exact cosine distance is correct and fast up to tens of thousands of chunks per document; cross-document search would add HNSW with `hnsw.iterative_scan`.
- **Lazy provider validation.** Every key is optional and checked when its provider is first used, so a pgvector-only deployment does not need a Pinecone key, and a misconfigured provider fails with a `503` naming the missing variable. The tradeoff is that misconfiguration shows up on first use rather than at boot.
- **Models.** Answers use `claude-haiku-5-5` at low effort: the task is reading excerpts and quoting them, where latency and cost matter more than deep reasoning. The judge uses `claude-opus-5-5`, a stronger model from a different tier than the answerer, which reduces the risk of a model grading its own style generously. Both are settings.
- **Heading-aware chunking instead of semantic chunking.** Semantic chunking needs embedding calls at ingest time and is not reproducible offline. Heading detection is free, deterministic, and gives citations a section path. The mock eval shows it is not automatically better, which is why the default follows the measurements.
- **Upload limits in layers.** `Content-Length` is checked before the body is read, the file is read in 1 MB pieces against `MAX_UPLOAD_MB`, and the bytes must start with `%PDF-` whatever the filename or content type claim. The multipart parser still spools the body before the handler runs, so production should also cap request size at the reverse proxy.
- **The in-memory store is per process.** Fine for tests, evals and demos; with several workers each one has its own copy.

## Project layout

```
app/
  main.py              app factory, middleware and error handler wiring
  config.py            pydantic-settings
  routes.py            /health, /upload, /chat, /query
  schemas.py           request/response models
  errors.py            error types and JSON envelope
  middleware.py        request IDs, access log, last-resort 500
  logging_config.py    text or JSON logs carrying the request ID
  services.py          lazily built providers
  rag/                 pdf.py, chunking.py, embeddings.py, generation.py, prompts.py, pipeline.py
  vectorstores/        base.py, pinecone.py, pgvector.py, memory.py
evals/
  run.py               runner and report
  dataset.jsonl        labelled questions
  metrics.py           precision@k, recall@k, MRR
  judge.py             rubric, Claude judge, lexical proxy
  corpus/              PDFs and their cleaned source text
tests/                 pytest suite (plus an optional pgvector integration test)
db/schema.sql          pgvector schema for manual setup
```

## Upgrading from the previous version

- Start the app with `uvicorn app.main:app` (was `main:app`).
- Supabase now goes through `DATABASE_URL` instead of `SUPABASE_URL` / `SUPABASE_KEY`; the table is `rag_chunks` (see `db/schema.sql`). `vector_store: "supabase"` is still accepted.
- `/upload` and `/chat` keep their request shapes. Upload rejections now use `415` for wrong file types and `422` for PDFs without text, and every error uses the envelope above.
