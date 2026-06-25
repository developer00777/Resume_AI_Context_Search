# Resume Intelligence — Update Log

## SOP Alignment: Phases 1–4

**Date:** 2026-06-24
**Scope:** Four gaps identified in the SOP-vs-current comparison. All implemented in one pass.

---

### Phase 1 — Query Parsing + Cypher Pre-Filter

**Problem:** `POST /api/search` was doing pure semantic search against the raw query string.
"Salesforce dev in Bangalore with 3+ years" and "Salesforce dev in Mumbai" returned identical
candidate pools — just differently ranked. The client's core complaint ("java returns all 200k
records undifferentiated") was not addressed by semantic search alone.

**What changed:**

- **New file:** `services/query_parser.py`
  - `QueryParser` class: single lightweight LLM call against the short query string (never the corpus).
  - Returns a `ParsedQuery` with: `location`, `min_yoe`, `max_yoe`, `must_skills`, `nice_skills`,
    `role`, `company_type`, `industries`, `semantic_query` (rewritten phrase for embedding).
  - Falls back to `ParsedQuery(semantic_query=original_query)` on any LLM failure — search
    degrades gracefully to pure semantic, never errors.
  - Strips markdown fences from LLM output; handles malformed JSON safely.

- **Updated:** `services/resume_service.py` — `search()` method is now a 5-step pipeline:
  1. `QueryParser.parse(query)` → `ParsedQuery`
  2. `_cypher_prefilter(parsed)` → list of matching candidate UUIDs
  3. `graphiti._search(semantic_query)` scoped to the filtered set
  4. Node list filtered to Cypher-matched candidates (non-candidate nodes kept for graph context)
  5. PII re-hydration (unchanged)

- **Cypher pre-filter** (`_cypher_prefilter`):
  - Location: `LOCATED_IN` edge → Location node name CONTAINS match
  - YoE: `duration.between(date(c.earliest_role_start), date()).years >= min_yoe`
  - Must-skills: `EXISTS { MATCH (c)-[:HAS_SKILL]->(s) WHERE toLower(s.name) CONTAINS skill }`
    — one EXISTS clause per must-skill (AND logic)
  - Role: `WORKED_AS` edge → Role node name CONTAINS match
  - Company type: `WORKED_AT` edge → Company node `company_type` property
  - Industries: `WORKED_AT` edge → Company node `industry` CONTAINS any listed industry
  - If filter returns 0 results: drops filter and falls back to full semantic (no empty results)
  - If Cypher fails: logs warning, falls back to full semantic (no crash)
  - Cap: 2000 candidate UUIDs max per pre-filter pass

- **API change:** `SearchResponse` now includes `parsed_filters` field — the structured
  filter object extracted from the query. Useful for the demo ("here's what we understood
  from your query") and for debugging.

**Before vs after:**
```
Query: "Salesforce developer in Bangalore with 3 years experience"

Before: Graphiti semantic search over all 200k candidates, ranked by embedding similarity.
        A Java dev in Mumbai with SF exposure could rank above a senior SF dev in Bangalore.

After:  Step 1 → { location: "Bangalore", min_yoe: 3, must_skills: ["salesforce"] }
        Step 2 → Cypher returns ~400 matching candidate UUIDs
        Step 3 → Graphiti semantic search over those 400 only
        Result: precision shortlist, not a ranked dump.
```

---

### Phase 2 — Cross-Candidate Edge Population (SKILL_CO_OCCURS_WITH + SIMILAR_TO)

**Problem:** The graph was a collection of isolated candidate stars. Every `HAS_SKILL`,
`WORKED_AT`, `WORKED_AS` edge pointed inward to one candidate. No edges crossed between
candidates or between skills across candidates. The graph had no intelligence that a
relational database couldn't provide.

**What changed:**

- **New file:** `services/edge_builder.py`
  - `EdgeBuilder` class — accepts the Neo4j driver that Graphiti already holds.
  - `build_all()` → runs both passes, returns a summary dict. Called after every ingest.
  - Never blocks the ingest response — fired as `asyncio.create_task`.

- **`SKILL_CO_OCCURS_WITH` pass:**
  - Cypher: for every pair of Skill nodes linked to the same Candidate, MERGE a
    `SKILL_CO_OCCURS_WITH` edge and increment `co_occurrence_count`.
  - Idempotent — safe to run multiple times.
  - Falls back to an episodic-edge variant if the primary Cypher fails.
  - Powers: adjacent skill discovery, D3 graph interest, "Salesforce devs also have Apex/SOQL".

- **`SIMILAR_TO` pass:**
  - Fetches all candidate `name_embedding` vectors from Neo4j (up to 5000).
  - Pure-Python cosine similarity (no numpy, no GDS dependency).
  - MERGES `SIMILAR_TO` edges for pairs above threshold (default 0.82), capped at top-10
    partners per candidate.
  - Powers: `GET /api/candidates/{name}/similar` now traverses real graph edges instead of
    running a semantic search query.

- **`services/resume_service.py` changes:**
  - `connect()` now instantiates `EdgeBuilder` alongside `QueryParser`.
  - `ingest_resume()` and `ingest_resumes_bulk()` both fire `asyncio.create_task(_run_edge_builder())`
    after the Graphiti episode write completes.
  - `_run_edge_builder()` catches all exceptions — a failed edge build never surfaces to the caller.
  - `find_similar_candidates()` now tries the `SIMILAR_TO` graph edge first, falls back to semantic
    search if no edges exist yet. Response includes `"source": "graph_edges"` or `"semantic_fallback"`.
  - `get_skill_pool()` now queries `SKILL_CO_OCCURS_WITH` edges for adjacent skills, returned as
    `adjacent_skills` list alongside the semantic candidate results.

- **New endpoint:** `POST /api/graph/rebuild-edges`
  - Triggers `EdgeBuilder.build_all()` synchronously and returns the summary.
  - Use after a large initial bulk load to compute edges over the full corpus at once
    (instead of waiting for incremental ingest to converge).

---

### Phase 3 — Pool Intelligence via Direct Cypher

**Problem:** `GET /api/pool/intelligence` was running a 200-result semantic search and
returning raw graph nodes/edges. Analytics questions ("which skills are thin?", "top locations?")
cannot be answered reliably by semantic search — a skill with 5 candidates and one with 500
rank similarly if the query embedding is close.

**What changed:**

- **`get_pool_intelligence()`** in `services/resume_service.py` replaced with 5 direct Cypher queries:
  1. **Top skills** — `HAS_SKILL` edges, group by skill name, order by candidate count DESC, LIMIT 30
  2. **Thin skills** — same query, order ASC, LIMIT 15 — supply gap view
  3. **Top locations** — `LOCATED_IN` edges, group by location name, LIMIT 20
  4. **Top companies** — `WORKED_AT` edges, group by company name, LIMIT 20
  5. **Total candidate count** — COUNT on Candidate-labeled nodes in `resume-pool` group

  Each query is wrapped in try/except — partial failures are logged but don't abort the response.
  If all five fail, falls back to the original semantic search approach.

- **Response shape change:** instead of `{nodes: [...], edges: [...]}`, returns:
  ```json
  {
    "top_skills": [{"skill": "java", "candidate_count": 1240}, ...],
    "thin_skills": [{"skill": "erlang", "candidate_count": 3}, ...],
    "top_locations": [{"location": "Bangalore", "candidate_count": 890}, ...],
    "top_companies": [{"company": "Infosys", "candidate_count": 320}, ...],
    "total_candidates": 4821
  }
  ```

- **`get_graph_for_visualization()`** still calls `get_pool_intelligence()` but `_format_for_visualization()`
  handles both response shapes (dict with `nodes/edges` or the new analytics dict).

---

### Phase 4 — `industries` Field on Candidate

**Problem:** The SOP's extraction schema includes `industries` (which sectors the candidate
has worked in). This is what enables queries like "ex-product company FinTech engineers".
The field was missing from `config/entity_types.py`.

**What changed:**

- **`config/entity_types.py`** — added `industries: Optional[list[str]]` to `Candidate`:
  ```python
  industries: Optional[list[str]] = Field(
      None,
      description=(
          "Industries the candidate has worked in, based on their employer history. "
          "Use standard sector names: FinTech, SaaS, Healthcare, E-commerce, "
          "EdTech, Consulting, Banking, Telecom, Manufacturing, Media, Government. "
          "Extract all that apply."
      )
  )
  ```
  Graphiti will now extract and store this list during ingest. The Cypher pre-filter
  in `_cypher_prefilter()` already handles `parsed.industries` via the `WORKED_AT →
  Company.industry` path — so once candidates are re-ingested with this field, the
  `industries` filter in `POST /api/search` will resolve against it automatically.

---

### Files Changed

| File | Change |
|------|--------|
| `services/query_parser.py` | **New** — LLM query parser, ParsedQuery |
| `services/edge_builder.py` | **New** — SKILL_CO_OCCURS_WITH + SIMILAR_TO edge builder |
| `services/resume_service.py` | **Updated** — search pipeline, pool intelligence, edge builder wiring |
| `config/entity_types.py` | **Updated** — `industries` field on Candidate |
| `api/models.py` | **Updated** — `parsed_filters` on SearchResponse |
| `api_server.py` | **Updated** — `parsed_filters` in search response, new `/api/graph/rebuild-edges` endpoint |

---

### New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/graph/rebuild-edges` | Trigger SKILL_CO_OCCURS_WITH + SIMILAR_TO edge recomputation |

### Changed Endpoint Behaviour

| Endpoint | Before | After |
|----------|--------|-------|
| `POST /api/search` | Pure semantic search | Parse → Cypher pre-filter → semantic in filtered set |
| `GET /api/pool/intelligence` | 200-result semantic search | 5 direct Cypher aggregation queries |
| `GET /api/candidates/{name}/similar` | Semantic search query | SIMILAR_TO graph edge traversal (semantic fallback if no edges) |
| `GET /api/skills/{skill}` | Semantic only | Semantic + `adjacent_skills` from SKILL_CO_OCCURS_WITH |

---

### What Remains (Post-Demo / Funded Build)

- Full-text index on raw resume text: `CREATE FULLTEXT INDEX resume_fulltext FOR (n:Entity) ON EACH [n.content]` — add to `connect()` in `resume_service.py`. Low priority — Graphiti's COMBINED_HYBRID_SEARCH_RRF already does BM25 over node names.
- Salesforce Apex bridge, Named Credential, Lightning component — SF dev task, not Python.
- On-server self-hosted embedding model (sovereignty) — already supported via `EMBEDDING_BASE_URL=http://localhost:11434/v1`, just a deployment config decision.
- `industries`-based Cypher pre-filter will become precise once re-ingested resumes carry the `industries` property. No code change needed — the filter already reads `Company.industry`.
