# RoleRadar

An AI-assisted opportunity-matching and adaptive interview-prep platform for technical college students — where Python makes every decision and the LLM only explains it.

## Problem

Technical students (SWE, quant, AI/ML, research) juggling multiple target opportunities lack a systematic way to determine:

- where they fit
- why
- what their biggest skill gaps are
- which opportunities deserve attention
- how to use limited interview-prep time

Most tools in this space are either a job board (no personalization) or a thin ChatGPT wrapper (a black-box score with no way to verify it, and no way to act on it). Neither answers "why did I get this score" or "what should I study tonight, given I have an interview in 6 days and 2 hours a day."

## Solution

RoleRadar ingests job postings and a candidate's resume, then runs a deterministic pipeline: normalize and deduplicate postings, filter out obviously-wrong matches cheaply before spending any LLM budget, extract structured requirements, and compute a six-component fit score entirely in Python. An evidence-grounded rationale (retrieval-augmented, never inventing a score) explains *why* — citing the actual resume/job-posting text behind each claim. Saved roles feed a cross-role skill-ROI analysis ("which single skill would most improve my position across every role I'm considering, and is it even worth learning"), and a separate, role-family-specific readiness score tracks *interview* preparedness, independent of job fit. A deadline-aware scheduler turns the resulting gaps into a day-by-day study plan that adapts — without discarding completed work — as diagnostic results come in.

The engineering thesis, repeated everywhere in the codebase: **Python decides, the LLM explains.** Every score, filter, gap, confidence value, and time allocation is deterministic, testable Python. The LLM's only job is turning unstructured text into structured data and writing prose about numbers it never gets to touch.

## Architecture

```
 RESUME                                   JOBS
   │                                        │
   ▼                                        ▼
 Candidate Extraction (LLM)          Concurrent Ingestion (ThreadPoolExecutor)
   │                                        │
   │                                        ▼
   │                                  Normalization (alias mapping → canonical Role)
   │                                        │
   │                                        ▼
   │                                  Deduplication (stable external_id)
   │                                        │
   │                                        ▼
   │                                  Rule Filtering (hard constraints → keyword overlap)
   │                                        │
   │                                        ▼
   │                                  Semantic Retrieval (*)
   │                                        │
   │                                        ▼
   │                                  Requirement Extraction (LLM, cached after first run)
   │                                        │
   └───────────────────┬───────────────────┘
                        ▼
                  Fit Scoring (6-component, deterministic)
                        │
                        ▼
                  Skill Gaps (deterministic)
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   RAG Evidence (Qdrant)        Readiness (role-family, no LLM)
   candidate + job chunks              │
          │                            │
          ▼                            │
   Rationale (LLM explains             │
   the already-computed score)         │
          │                            │
          ▼                            │
      Favorites ──► Cross-Role Analysis / Skill ROI
          │                            │
          └─────────────┬──────────────┘
                         ▼
              Adaptive, Deadline-Aware Prep Planning
              (diagnostic results → replan remaining
               time, completed tasks preserved)
```

`(*)` **Semantic Retrieval as a pre-LLM filter stage is designed but not wired in yet** — its counter field exists end to end in the metrics schema and always reports 0 today. The Qdrant/embedding infrastructure it would use is real and already running, just further downstream (RAG Evidence, for rationale generation) rather than as a filter ahead of requirement extraction. See **Future Improvements**.

Two front ends sit on top of the same `backend/services/*` layer and never contain business logic themselves:

- **Streamlit app** (`app.py` + `pages/`) — the primary UI: Discover, Favorites, Applications, Skill Gaps, Interview Prep, Analytics, Role Analysis.
- **FastAPI app** (`backend/main.py` + `backend/api/`) — a REST surface over the same services (candidate parsing, roles, favorites, applications, plans, assessments, analytics), for programmatic or future non-Streamlit clients.

```
backend/
  candidate/   resume parsing (PyMuPDF) + LLM profile extraction + skill normalization
  ingestion/   source adapters, concurrent fetch, normalization, dedup, filtering
  llm/         OpenAI client wrapper, prompts, requirement extraction, evidence-grounded rationale
  matching/    deterministic fit scoring, skill gaps, confidence, cross-role ROI
  planning/    readiness, deadline-aware scheduling, adaptive replanning
  rag/         chunking, embeddings, Qdrant vector store, retrieval
  db/          one module per table; idempotent upserts only
  services/    orchestration layer both front ends call into
  api/         FastAPI routers + Pydantic schemas + error handlers
  utils/       config, hashing, logging, pipeline metrics
pages/         Streamlit multipage app
sql/           schema.sql (re-runnable, additive migrations)
scripts/       demo_end_to_end.py — full-pipeline integration script
tests/         563 tests, offline (fake Postgres/Qdrant/OpenAI clients)
```

## Tech Stack

Python 3.11+ · Streamlit · FastAPI · Supabase (PostgreSQL) · Qdrant · OpenAI (chat + embeddings) · Pydantic · `concurrent.futures.ThreadPoolExecutor` · PyMuPDF · pytest · GitHub Actions

Also in real use: Plotly (Analytics charts), `python-dotenv` (config loading), `httpx` (FastAPI's test client).

## Core Engineering Decisions

**1. Python decides, the LLM explains.** No LLM call ever returns a score, a filter decision, a confidence value, or a time allocation. `backend/matching/scorer.py`'s fit score, `backend/matching/gaps.py`'s skill gaps, `backend/planning/readiness.py`'s readiness, and `backend/planning/scheduler.py`'s minute-by-minute allocation are all plain arithmetic over structured Pydantic models. The LLM extracts structured data from resumes/postings (`backend/candidate/profile.py`, `backend/llm/extract_requirements.py`) and writes narrative text about numbers that are handed to it as fixed, already-final context (`backend/llm/rationale.py`) — the response schemas it fills in structurally have no field for a score, so it's not just a convention, it's enforced by the type the LLM is constrained to return.

**2. Idempotent upserts everywhere.** Every write path is safe to call twice. Real UNIQUE constraints back every upsert (see **Database** below), and `backend/db/upserts.py` centralizes the `ON CONFLICT` pattern so ingesting the same job twice, re-analyzing the same role, or re-saving the same favorite never creates a duplicate row.

**3. Staged filtering before the LLM ever runs.** `backend/matching/filters.py` runs cheap, deterministic hard constraints (deadline, location, seniority, major, ...) first, then a lexical keyword/skill-overlap pass — both pure Python, zero API cost — before a posting reaches `extract_role_requirements()`. `backend/utils/metrics.py`'s `llm_avoidance_rate()` measures exactly what fraction of eligible postings never needed an LLM call, from real recorded counts, never a guess. (A third stage, semantic/embedding retrieval, has its counter field wired through the schema and always reports 0 today — the vector infrastructure it would use already exists for evidence retrieval, but it isn't yet plugged in as a pre-LLM filter.)

**4. Deterministic, six-component fit scoring.** `calculate_fit_score()` combines technical, experience, coursework, domain, interest, and constraint fit into one weighted 0–100 score (weights below). Confidence is deliberately kept as a *separate* number rather than folded into the score, so an under-evidenced-but-plausible estimate never looks artificially worse than a well-evidenced one at the same claimed level.

**5. Evidence-grounded RAG, retrieval enforced before generation.** `backend/llm/rationale.py` structurally cannot call the LLM before `_retrieve_evidence()` runs — the generation function's only input is that retrieval's output. Every claim in a rationale is traceable to an actual retrieved chunk; a skill with no evidence gets `insufficient_evidence=True` rather than a fabricated explanation, even if the LLM claims otherwise (Python overrides it when zero chunks came back).

**6. Fit ≠ Readiness — two different questions, two separate modules.** Fit (`matching/scorer.py`) asks "is this candidate a good match for this *role*." Readiness (`planning/readiness.py`) asks "is this candidate prepared to *interview* right now," scored against a curated, role-family-specific interview-topic taxonomy (quant vs. SWE vs. general), independent of any one job posting. A candidate can have high fit and low readiness, or the reverse — conflating them would hide that distinction.

**7. Adaptive, deadline-aware scheduling that never rewrites history.** `backend/planning/scheduler.py` builds an initial plan from remaining time × ranked priorities. `backend/planning/adaptive.py` doesn't mutate that plan when a diagnostic result comes in — it produces a new, versioned revision: completed tasks are carried forward *unchanged*, only the unallocated remaining time is redistributed against updated readiness.

**8. Favorites are the input to cross-role analysis, not every ingested role.** `backend/matching/cross_role.py`'s skill-ROI and side-by-side comparison deliberately scope to a candidate's *saved* roles (2–4 for comparison), weighted by favorite priority (dream/high/interested/backup) — "which skill matters across the roles I actually want," not a diluted average across everything in the database.

**9. Concurrency measured, not assumed.** `backend/ingestion/concurrent.py` runs the exact same fetch logic serially and concurrently (via `ThreadPoolExecutor`) purely to produce an honest, apples-to-apples wall-clock speedup number — nothing is hardcoded. (Building this surfaced a real bug worth knowing about: `as_completed()` yields already-finished futures, so calling `.result(timeout=...)` on them can never actually time out; the fix iterates futures in submission order instead.)

## Database

Supabase/PostgreSQL, one module per table under `backend/db/`, every table's idempotency guaranteed by a real `UNIQUE` constraint — never by application-level "check first" logic alone:

| Table | Unique on | Why |
|---|---|---|
| `roles` | `external_id` | Re-ingesting the same posting (derived from a deterministic hash of company+title+location+url when the source has no stable ID) always resolves to one row. |
| `candidate_skills` | `(user_id, normalized_skill_name)` | Re-estimating a skill (new resume, new diagnostic) updates in place. |
| `role_requirements` | `(role_id, normalized_skill_name)` | Re-extracting a role's requirements updates in place, never duplicates. |
| `fit_scores` | `(user_id, role_id)` | One current score per candidate/role pair. |
| `favorites` | `(user_id, role_id)` | Re-saving a favorite is a no-op — priority/notes are preserved, not reset (a blind upsert here would silently clobber a user's chosen priority). |
| `applications` | `(user_id, role_id)` | Partial updates preserve every field the caller didn't touch — implemented as an explicit update-if-exists-else-insert, since a plain `ON CONFLICT` upsert would need every column supplied every time. |
| `role_rationales` | `(user_id, role_id)` | Cached rationale, keyed with a hash of the exact skills+requirements it was generated from — reused as-is until an input actually changes. |
| `candidate_profiles` | `(user_id)` | One profile row per candidate. |

`study_plans` takes a different approach: no plan is ever deleted or mutated once written. A revision inserts a new row and flips the prior "current" row's `is_current` flag to `false`, so the full version history stays queryable. `ingestion_runs` is deliberately append-only (no unique constraint) — every pipeline execution is a genuinely new audit-log event.

## RAG

- **Chunking** (`backend/rag/chunking.py`) — builds labeled `EvidenceChunk`s directly from already-structured Pydantic models (a candidate's skills/projects/experience, a role's requirements), not a naive re-split of raw text — so every chunk carries correct metadata (which skill, which project) for free.
- **Embeddings** (`backend/rag/embeddings.py`) — a small `EmbeddingProvider` interface; the only implementation today wraps OpenAI's embedding API, swappable later without touching chunking or storage.
- **Qdrant** (`backend/rag/vector_store.py`) — three collections (`candidate_evidence`, `role_content`, `prep_resources` — reserved, unused). Point IDs are deterministic (`UUID5` of user+role+source+skill+text), so re-indexing the same logical chunk overwrites in place instead of accumulating duplicates. Querying a collection that was never created (a brand-new user, nothing indexed yet) is treated as "no results," not an error.
- **Retrieval** (`backend/rag/retrieval.py`) — per-skill (`retrieve_skill_evidence`) and per-dimension (`retrieve_candidate_evidence` / `retrieve_role_content`) top-k lookups, always scoped by metadata filter (user_id, role_id, skill) so one candidate's evidence never leaks into another's results.
- **Rationale** (`backend/llm/rationale.py`) — retrieval always runs first; the LLM narrates six fit dimensions (technical/experience/coursework/domain/interest/constraints) plus per-skill assessments, each confidence value either copied from the already-computed gap or derived from real retrieval-similarity scores — never invented by the model.

## Matching Algorithm

```
overall_score = 0.35·technical + 0.20·experience + 0.10·coursework
              + 0.15·domain    + 0.10·interest    + 0.10·constraints
```

- **Technical**: importance-weighted average of per-requirement satisfaction (`min(candidate_level / target_level, 1.0)`); a required requirement counts 1.5× as much as an equally-important optional one.
- **Skill gap**: `raw_gap = max(target_level − candidate_level, 0)`; `weighted_gap = raw_gap × (importance / 10)`.
- **Experience / coursework / domain / interest / constraints**: documented heuristics (internship/project counts, coursework keyword coverage, domain word-overlap, location/remote conflicts) that degrade to a *neutral* score rather than zero when the underlying data doesn't exist yet — an under-specified input never arbitrarily tanks a candidate's score.
- **Confidence** is aggregated separately (`backend/matching/confidence.py`) and never folded into the score itself.

## Prep Algorithm

```
priority_score = gap × requirement_importance × interview_topic_weight
                × confidence_adjustment × favorite_priority_multiplier
```

`confidence_adjustment` floors at 0.5 (never zero — missing evidence is itself a reason to study something, not a reason to skip it) and reaches 1.0 at full confidence. Available minutes are split proportional to `priority_score` (the lowest-priority item absorbs any rounding remainder, so the total always sums exactly). A diagnostic result (`backend/planning/adaptive.py`) doesn't regenerate the whole plan — it recomputes readiness with the new evidence folded in (diagnostics outweigh resume-only evidence 3:1 by design), keeps every completed task exactly as it was, and reschedules only the unallocated remaining time.

## Favorites / Skill ROI

Cross-role analysis (`backend/matching/cross_role.py`) runs only over a candidate's *saved* roles:

```
raw_roi = average_gap × roles_requiring_it × average_importance × weighted_priority
          / estimated_learning_cost_hours
```

`weighted_priority` is the mean `FAVORITE_PRIORITY_MULTIPLIER` (dream=3.0, high=2.0, interested=1.0, backup=0.5) across the roles that need that skill — so a skill only one *backup* role needs never outranks one three *dream* roles need. `raw_roi` is then normalized 0–100 relative to the best-scoring skill in that candidate's current favorite set — a ranking aid for *this* candidate's *current* choices, explicitly not a portable or absolute measurement (documented as a heuristic, not dressed up as more precise than it is).

## Metrics

No fabricated numbers, ever — `PipelineMetrics.llm_avoidance_rate()` returns `None` rather than a fake percentage when there's nothing to compute a rate over, and `ConcurrencyMetrics.speedup()` is always the ratio of two *measured* wall-clock durations, never a hardcoded multiplier. This development environment has no Supabase project connected, so there's no live production pipeline history to report here. What's concretely true right now:

- **563 passing tests** (see **Tests** below), including ones that empirically measure concurrent-vs-serial ingestion speedup with real threads and real timing, not simulated numbers.
- The Analytics page (`pages/6_Analytics.py`) and `GET /analytics/pipeline` render exactly what's described above from real `ingestion_runs` rows once a database is connected — including an explicit empty state, never a placeholder chart, when none exist yet.

## Running Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your own credentials
```

Streamlit app:

```bash
streamlit run app.py
```

FastAPI app:

```bash
uvicorn backend.main:app --reload
```

Full-pipeline integration script (works with or without credentials — see the script's own docstring for exactly what falls back to a local/mocked path when Supabase/OpenAI/Qdrant aren't configured):

```bash
python scripts/demo_end_to_end.py
```

## Environment Variables

Set in `.env` (see `.env.example` — names only, no values committed):

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

Every one of these is optional at startup — each missing credential degrades a specific, clearly-labeled part of the app (e.g. Streamlit's "Demo mode," or a 503 from the FastAPI layer) rather than crashing.

## Tests

```bash
pytest
```

**563 tests, all passing**, and all fully offline — no real Supabase, OpenAI, or Qdrant call happens in the suite. Persistence tests run against an in-memory fake Postgres client (`tests/_fake_supabase.py`) with real `ON CONFLICT`/idempotency semantics; LLM and Qdrant calls are monkeypatched at the client boundary; four Streamlit pages are exercised end-to-end via `streamlit.testing.v1.AppTest` against that same fake database. Coverage spans normal paths and a deliberate edge-case audit — duplicate ingestion, empty/malformed PDFs, zero skill overlap, interview-today/interview-passed/zero-minutes planning, one ingestion source failing, a missing Qdrant collection, and more.

## Future Improvements

- Real authentication — today every request (Streamlit and API) resolves to a single demo/dev user (`backend/db/users.py`) by design, not yet swapped for real auth.
- A Streamlit resume-upload flow — `backend/services/candidate.py` and `POST /candidate/parse` already do the real extract-and-persist work; no page wires a file uploader to it yet.
- Real job-source adapters — `backend/ingestion/jobs.py` ships demo adapters; a production run needs real API/board integrations behind the same `SourceAdapter` interface.
- The semantic-retrieval filter stage — the counter exists end to end and always reports 0; plugging in an actual pre-LLM embedding-similarity filter is the next natural step in `backend/matching/filters.py`.
- LLM call timeout/retry handling — currently a raw SDK timeout propagates as-is; there's no retry/backoff policy yet (a deliberate scope decision, documented via a test that pins down the current behavior rather than silently hoping it doesn't happen).
- Broader Streamlit UI test automation — `AppTest` coverage today is limited to four empty-state scenarios; interactive flows (button clicks, form submissions) are currently verified manually.
