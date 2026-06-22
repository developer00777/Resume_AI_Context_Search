# Resume Intelligence

Semantic search + knowledge graph for resume data.
Built on ChampGraph's Graphiti + Neo4j/FalkorDB engine, re-pointed at the resume domain.

## What this is

A FastAPI service that ingests resume text, runs LLM-based structured extraction via Graphiti,
stores everything (vectors + graph edges + temporal state) in one Neo4j/FalkorDB instance,
and exposes hybrid semantic + keyword search over the candidate pool.

No Postgres. No pgvector. One store.

## Architecture

```
Resume text (from temp server / Salesforce export)
        │
        ▼
POST /api/ingest/bulk
        │
        ▼
  models/resume.py  ── Resume.to_episode_content() → structured text
        │
        ▼
  services/resume_service.py  ── ResumeService.ingest_resume(s)_bulk()
        │
        ▼
  Graphiti  ── LLM extraction → entity/edge nodes → embedding on nodes
        │
        ▼
  Neo4j / FalkorDB  ── ONE store: vectors + graph + temporal + full-text index
        │
        ▼
POST /api/search  ── Graphiti semantic search + Cypher structured filter
GET  /api/candidates/{name}
GET  /api/skills/{skill}
GET  /api/pool/intelligence
GET  /api/graph  ── D3 visualization data
```

## Project structure

```
resume-intelligence/
├── CLAUDE.md                   # This file
├── api_server.py               # FastAPI server (port 8080)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── Dockerfile                  # Container
├── docker-compose.yml          # Neo4j + API server
│
├── api/
│   ├── auth.py                 # X-API-Key middleware (unchanged from ChampGraph)
│   └── models.py               # Request/response models (resume domain)
│
├── config/
│   ├── settings.py             # Env var config (simplified — no Gmail/MS365)
│   ├── entity_types.py         # Graph nodes: Candidate, Skill, Role, Company, Location, Education, Certification
│   └── edge_types.py           # Graph edges: HAS_SKILL, WORKED_AT, WORKED_AS, LOCATED_IN, STUDIED_AT, HOLDS_CERTIFICATION, SKILL_CO_OCCURS_WITH, SIMILAR_TO
│
├── models/
│   └── resume.py               # Resume Pydantic model + to_episode_content()
│
├── services/
│   └── resume_service.py       # ResumeService: connect, ingest, search, graph queries
│
├── visualization/
│   └── index.html              # D3.js graph viewer (unchanged from ChampGraph)
│
└── tests/                      # Existing ChampGraph tests (to be updated)
```

## Graph schema

### Entity types (nodes) — config/entity_types.py

| Entity | Key fields | Purpose |
|--------|-----------|---------|
| Candidate | full_name, source_id, earliest_role_start, current_location | The person |
| Skill | normalized, evidence, recency | A skill (canonical lowercase) |
| Role | title, normalized | A job title |
| Company | company_name, industry, company_type | An employer |
| Location | city, country | A geographic location |
| Education | degree, institution, graduation_year | A qualification |
| Certification | cert_name, issuer | A professional certification |

### Edge types — config/edge_types.py

| Edge | Source → Target | Key properties |
|------|----------------|----------------|
| HAS_SKILL | Candidate → Skill | proficiency, recency, evidence, years_used |
| WORKED_AT | Candidate → Company | start, end, title, tenure_months |
| WORKED_AS | Candidate → Role | start, end, company |
| LOCATED_IN | Candidate → Location | is_current |
| STUDIED_AT | Candidate → Education | degree, graduation_year |
| HOLDS_CERTIFICATION | Candidate → Certification | obtained_year, is_active |
| SKILL_CO_OCCURS_WITH | Skill → Skill | co_occurrence_count |
| SIMILAR_TO | Candidate → Candidate | similarity_score |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Liveness — no auth required |
| POST | /api/ingest | Ingest a single resume |
| POST | /api/ingest/bulk | Bulk ingest up to 500 resumes |
| POST | /api/search | Hybrid semantic + keyword search |
| GET | /api/candidates/{name} | Full candidate graph context |
| GET | /api/candidates/{name}/similar | Find similar candidates |
| GET | /api/skills/{skill} | All candidates with a skill + adjacent skills |
| GET | /api/pool/intelligence | Skill/location/role distribution across pool |
| GET | /api/utils/yoe | Compute live years of experience from date anchor |
| GET | /api/graph | Full graph data for D3 visualization |

## Key design decisions

1. **One store** — Graphiti on Neo4j/FalkorDB handles vectors, graph, temporal, full-text. No Postgres, no pgvector.
2. **Single group_id** — All resumes go into `resume-pool` so cross-candidate graph queries (SIMILAR_TO, SKILL_CO_OCCURS_WITH) work.
3. **Extraction at ingest** — LLM runs once per resume at ingest time. Search never re-embeds the corpus.
4. **Live YoE** — `earliest_role_start` stored on employment edge by Graphiti temporal layer. `current_years_experience()` in `resume_service.py` computes live.
5. **Sovereign by design** — Point `EMBEDDING_BASE_URL` to Ollama running on the client's server. Resume text never leaves their box.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| OPENAI_API_KEY | Yes | OpenRouter key (or OpenAI) |
| OPENAI_BASE_URL | Yes | LLM endpoint |
| MODEL_NAME | Yes | Extraction model (recommend anthropic/claude-sonnet-4) |
| NEO4J_URI | Yes | Neo4j bolt URI |
| NEO4J_USER | Yes | Neo4j user |
| NEO4J_PASSWORD | Yes | Neo4j password |
| EMBEDDING_MODEL | No | Defaults to openai/text-embedding-3-small |
| EMBEDDING_BASE_URL | No | Override to Ollama for on-server sovereignty |
| API_KEY | No | X-API-Key auth header value (disabled if unset) |
| API_PORT | No | Defaults to 8080 |

## Quick start

```bash
# 1. Copy and fill env
cp .env.example .env

# 2. Start Neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword neo4j:5.26.2

# 3. Install deps
pip install -r requirements.txt

# 4. Start server
python api_server.py

# 5. Ingest a resume
curl -X POST http://localhost:8080/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_id": "SF001", "full_text": "John Smith...", "candidate_name": "John Smith"}'

# 6. Search
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "backend engineer with Java and Spring Boot in Bangalore", "num_results": 20}'

# 7. Open graph viewer
open visualization/index.html
```
