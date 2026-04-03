Last Updated: 2026-04-03

## Snapshot
This repository now has a working multilingual PDF ingestion backend and an implemented retrieval/query backend for grounded multimodal answers. The frontend is still template-level.

Implemented and verified:
- page-level ingestion routing between native, OCR, and Docling
- doc-id-scoped reruns and cleanup
- OCR line merging to reduce chunk fragmentation
- image extraction, captioning, and image-text linking
- batched Qdrant upserts
- Cloudinary uploads with doc-id-scoped asset IDs
- per-page logs and run-state checkpoints
- hybrid retrieval over `text_chunks` and `image_chunks`
- multilingual reranking with conditional image/table evidence selection
- context assembly with stable citation IDs
- Groq-backed answer engine with sync and streaming paths
- FastAPI `/query`, `/query/stream`, and `/health` endpoints
- retrieval/API unit tests covering sparse-link rescue, conditional tables, missing-image guards, and SSE contract shape

The most important current guarantee is still rerun identity:
- same PDF name + same file size + same resolved absolute path => same `doc_id`
- move or rename the file => new `doc_id`
- reruns delete the prior Qdrant points and Cloudinary assets for that `doc_id` before re-upserting
- output folders and logs are doc-id scoped so same-named PDFs in different folders do not collide

Still incomplete:
- frontend is not wired to the backend
- API auth/rate limiting is not implemented
- no dedicated image-serving endpoint is implemented because the API returns Cloudinary URLs directly

## Source Of Truth
This handoff is aligned to code reality in the repository, not to earlier planning notes.

Core ingestion files:
- backend/scripts/ingest_documents.py
- backend/ingestion/page_classifier.py
- backend/ingestion/native_pdf_parser.py
- backend/ingestion/ocr_pipeline.py
- backend/ingestion/docling_parser.py
- backend/ingestion/hybrid_page_repair.py
- backend/ingestion/page_model_builder.py
- backend/ingestion/page_layout.py
- backend/ingestion/noise_detector.py
- backend/ingestion/image_extractor.py
- backend/ingestion/image_text_linker.py
- backend/ingestion/section_detector.py
- backend/ingestion/shloka_detector.py
- backend/ingestion/chunker.py
- backend/ingestion/qdrant_mapper.py
- backend/ingestion/cloudinary_uploader.py
- backend/vector_db/qdrant_client.py
- backend/embeddings/text_embedder.py
- backend/embeddings/image_embedder.py

Core query-serving files:
- backend/retrieval/hybrid_search.py
- backend/retrieval/reranker.py
- backend/rag/context_builder.py
- backend/rag/query_engine.py
- backend/api/models.py
- backend/api/main.py

Supporting docs:
- backend/docs/multilingual_pdf_rag_architecture_report.md
- backend/docs/multilingual_pdf_rag_ingestion_execution_spec.md

## Real Query Pipeline In Code
The query-serving path now runs as follows:

1) Query normalization and intent detection
- `HybridSearcher.build_query_bundle()` cleans the query, creates a normalized query using `DiacriticNormalizer`, and infers intent as `general`, `visual`, `table`, `shloka`, or `definition`.
- Public filters currently supported: `doc_id`, page range, `languages`, `scripts`, and `chunk_types`.
- `section_path` is deliberately not exposed as a v1 public filter because live heading quality is uneven.

2) Retrieval
- Text retrieval runs against `text_chunks` using hybrid dense + sparse search over the original query.
- If normalized query differs from original query, a second text retrieval pass runs and results are deduplicated by `chunk_id`.
- Image retrieval runs separately against `image_chunks` with dense search and always excludes `image_type=decorative`.
- Retrieval keeps tables as normal text evidence by preserving `chunk_type=table_text`.

3) Sparse-link rescue
- If a strong text chunk carries `image_ids`, linked image points are fetched explicitly.
- If linkage is absent or sparse, same-page and adjacent-page image rescue can run using `doc_id` + page proximity.
- This fallback exists because live image linkage is present but not dense across the whole corpus.

4) Reranking and evidence selection
- `CandidateReranker` reranks the merged top-24 candidates with `BAAI/bge-reranker-v2-m3`.
- Text rerank content includes heading and `table_markdown` when present.
- Image rerank content includes caption, labels, surrounding text, and section path.
- Light deterministic boosts apply after rerank for:
  - table intent on `table_text`
  - shloka intent on `shloka`
  - linked images for visual intent
  - `page_bridge` downweight when both source chunks are already stronger
- Final evidence caps:
  - up to 6 text citations
  - up to 2 image candidates
  - up to 2 table candidates

5) Context assembly and answer gating
- `ContextBuilder` assigns stable IDs:
  - `C#` for citations
  - `F#` for image cards
  - `T#` for table cards
- Prompt assembly uses a deterministic token budget (default 2800 tokens).
- Images are only returned when they have a valid `image_url` and survive evidence selection.
- Tables are conditional:
  - table cards are returned only when structure is meaningful and the table is genuinely useful for the question
  - otherwise table evidence is still available through citations and answer text without emitting a `tables` card

6) Answer synthesis
- `QueryEngine` owns the full orchestration: query bundle -> retrieval -> rerank -> context -> answer.
- `LLMClient` abstraction isolates provider choice.
- `GroqLLMClient` is the current implementation.
- Default model is `llama-3.3-70b-versatile`, overridable by `GROQ_MODEL`.
- If evidence is weak, the engine skips synthesis and returns a grounded fallback response.
- If the Groq key is missing, the engine returns evidence-backed fallback text instead of hallucinating.

7) API transport
- `POST /query` returns the full response synchronously.
- `POST /query/stream` streams answer tokens and then emits one final metadata event.
- `GET /health` checks Qdrant reachability and LLM readiness.

## Data And Response Contract Snapshot
Qdrant payloads remain aligned to ingestion reality.

Text chunks:
- point ID pattern: `{doc_id}:p{page_start}-{page_end}:{chunk_type}:{chunk_index}`
- important retrieval fields: `chunk_id`, `doc_id`, `source_file`, `page_start`, `page_end`, `page_numbers`, `chunk_type`, `text`, `normalized_text`, `section_path`, `heading_text`, `languages`, `scripts`, `layout_type`, `route`, `ocr_source`, `ocr_confidence`, `image_ids`, `bridge_source_chunk_ids`, `table_id`, `table_rows`, `table_markdown`

Image chunks:
- point ID pattern: `{doc_id}:p{page_number}:img:{figure_index}`
- important retrieval fields: `image_id`, `doc_id`, `source_file`, `page_number`, `image_type`, `caption`, `labels`, `surrounding_text`, `section_path`, `linked_chunk_ids`, `cloudinary_public_id`, `image_url`

Vector model and shapes:
- text: dense + sparse lexical vectors from BGE-M3
- image: dense vectors from BGE-M3 over caption/context text
- Qdrant dense dimension is 1024

Public API request shape:
- `query`
- optional `doc_id`
- optional `page_start`, `page_end`
- optional `languages`
- optional `scripts`
- optional `chunk_types`
- optional `include_debug`

Public API response shape:
- `answer`
- `citations`
- `images`
- `tables`
- `enough_evidence`
- `query_intent`
- `model`
- `timings`
- optional `debug`

SSE contract for `POST /query/stream`:
- `token`
  - incremental answer text
- `final`
  - full response payload equivalent to `/query`
- `error`
  - structured error payload

## Environment And Runtime
Mandatory env keys for end-to-end backend query serving:
- QDRANT_URL
- QDRANT_API_KEY
- CLOUDINARY_CLOUD_NAME
- CLOUDINARY_API_KEY
- CLOUDINARY_API_SECRET
- GROQ_API_KEY for LLM answer generation

Important runtime notes:
- use the backend venv interpreter explicitly
- current imports assume the backend directory is on `PYTHONPATH` or is the working directory when running backend modules directly

## Tests Available
Current backend tests:
- backend/tests/test_ingestion_modules.py
- backend/tests/test_qdrant.py
- backend/tests/test_retrieval_query_api.py

What the new retrieval/API tests cover:
- multilingual retrieval intent scaffolding
- linked-image rescue and page-proximity fallback
- conditional table display
- missing-image-card guard when `image_url` is absent
- query engine fallback behavior when LLM generation is unavailable
- API `/query` and `/query/stream` contract shape

Verified during this update:
- `backend\venv\Scripts\python.exe -m pytest tests\test_retrieval_query_api.py -q`

## Known Gaps And Risks
1) `doc_id` stability is machine-path dependent
- `doc_id` still includes resolved absolute path in its hash input
- the same PDF moved to a different machine/path can produce a different `doc_id`

2) Frontend is still template state
- frontend/app/page.tsx is still the default Next app starter page
- no backend integration yet

3) Query API still lacks operational hardening
- no auth
- no rate limiting
- no request logging / tracing layer beyond what the query engine returns in debug

4) Reranker and embeddings depend on local model availability
- first-time model downloads may be slow
- machines without enough memory may need smaller batch sizes or alternate settings

5) Section quality is not clean enough for public filtering
- section paths are useful as internal ranking context
- they should not be treated as a trusted user-facing filter until heading quality improves

6) Image usefulness varies by corpus quality
- many images have strong URLs and valid payloads
- some still rely mostly on surrounding text because captions/labels are sparse

## Recommended Resume Commands
Use from repository root in PowerShell:

1. Activate backend environment
   - .\backend\venv\Scripts\Activate.ps1

2. Run ingestion for one PDF
   - python backend\scripts\ingest_documents.py --pdf <pdf_path>

3. Run the new retrieval/API tests
   - cd backend
   - .\venv\Scripts\python.exe -m pytest tests\test_retrieval_query_api.py -q

4. Run the FastAPI service
   - cd backend
   - .\venv\Scripts\python.exe -m uvicorn api.main:app --reload

5. Exercise the sync endpoint
   - POST `http://127.0.0.1:8000/query`

6. Exercise the streaming endpoint
   - POST `http://127.0.0.1:8000/query/stream`

## Immediate Next Work Items
1. Wire the frontend to `/query` and `/query/stream`.
2. Add auth and rate limiting to the API.
3. Add broader integration tests against live Qdrant + Groq.
4. Decide whether to expose retrieval-only mode alongside answer-producing mode.

## Maintenance Rule
Keep this file as the single-source current snapshot.
When updating, replace outdated statements instead of appending historical change logs.
