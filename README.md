<<<<<<< HEAD
# RoleRadar

AI-powered opportunity matching and adaptive interview-readiness platform for
technical college students (SWE, quant, AI/ML, and research roles).

RoleRadar is designed as a technically deep engineering project, not a thin
LLM wrapper. The architectural principle throughout the codebase is:

- **Python decides**: filtering, deduplication, scoring, skill-gap
  calculation, confidence estimation, prep-time allocation, cross-role ROI,
  and all application/database state are deterministic Python logic.
- **The LLM explains**: it extracts structured data from unstructured text
  (resumes, job descriptions), generates evidence-backed rationales for
  scores already computed, and writes study-task wording after the
  time allocation has already been decided algorithmically.

## Status

The PostgreSQL persistence layer (schema + idempotent upsert helpers) is
implemented. Resume parsing, ingestion, scoring, RAG, and planning are
not yet implemented.

## Tech Stack

- **Language**: Python 3.11+
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Relational DB**: Supabase PostgreSQL
- **Vector DB**: Qdrant Cloud
- **LLM**: OpenAI API
- **Validation**: Pydantic
- **Concurrency**: concurrent.futures.ThreadPoolExecutor
- **Data**: pandas
- **PDF parsing**: PyMuPDF
- **HTTP**: httpx
- **Testing**: pytest

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and fill in your own credentials:

```bash
cp .env.example .env
```

## Database Setup (Supabase)

1. Create a project at [supabase.com](https://supabase.com) (free tier is fine).
2. In the project dashboard, open **SQL Editor** -> **New query**, paste the
   contents of [`sql/schema.sql`](sql/schema.sql), and run it. It is safe to
   re-run.
3. Find `SUPABASE_URL` and the API key under **Project Settings -> API**:
   - `SUPABASE_URL` = "Project URL"
   - `SUPABASE_KEY` = the `service_role` secret key for local/dev use (never
     ship this to a browser-facing client)
4. Put both values in `.env`:

   ```
   SUPABASE_URL=https://<your-project-ref>.supabase.co
   SUPABASE_KEY=<your-service-role-key>
   ```

There is no authentication system yet. Every table hangs off a minimal
`users` row (id, email) so a single development/demo user can be created
manually for now; real auth (e.g. Supabase Auth) can replace it later
without changing the other tables.

If `SUPABASE_URL`/`SUPABASE_KEY` are unset, `backend/db/client.py` raises
`SupabaseNotConfiguredError` rather than failing silently - the rest of the
codebase and its unit tests do not require live credentials.

## Running the app

### Streamlit frontend

```bash
streamlit run app.py
```

### FastAPI backend

```bash
uvicorn backend.main:app --reload
```

The backend currently exposes a single health check endpoint at
`GET /health`.

## Running tests

```bash
pytest
```

## Project Structure

See `backend/`, `pages/`, and `tests/` for the module layout, and
`sql/schema.sql` for the database schema. Backend modules outside
`backend/db/` and `backend/utils/hashing.py` currently contain only a
docstring describing their intended responsibility; implementation will
be added incrementally.
=======
# RoleRadar
>>>>>>> c765b605b59c83f6cd4b1b83a6c27dfff053ee31
