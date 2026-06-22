---
created: "2026-06-22"
updated: "2026-06-22 (revised: single unified phase, Graphiti + Neo4j/FalkorDB, no pgvector/Postgres)"
tags:
  - effort
  - context
  - ai-infrastructure
  - knowledge-graph
company: "[[Cirralogix]]"
status: active
type: reference
related:
  - "[[Cirralogix]]"
  - "[[Recruit Champ]]"
  - "[[ChampGraph]]"
  - "[[Cirralogix SF-PC Migration — Reference Doc]]"
---

# Cirralogix Resume Intelligence: Semantic Search + Knowledge Graph Architecture and MVP Plan

> [!info] Purpose
> Decode of the recorded call on the [[Cirralogix]] resume-search problem, plus the target architecture and a 6-hour-to-1-day MVP we can showcase. The end goal from the call: a rough but showable prototype that proves we can fix search precision and graph relationships in one build. Owner: [[Sreedeep Surapaneni]]. Audience for this doc: the build team and the technical partner on the call.

---

## 0. TL;DR (one screen)

1. **Two concrete failures** in the client's current Salesforce parser: stale years-of-experience and weak skills extraction. The first is a deterministic bug we can fix in an afternoon with zero AI. The second needs better extraction at ingest plus semantic search at query time.
2. **What is already live:** data pulled out of Salesforce to a temporary server, SQL plus Python search component, phrase matching with pagination (50 per page). That is keyword matching. It works but it is dumb.
3. **The upgrade is one phase, not two.** Semantic search and the knowledge graph are built together on a single store — Graphiti on Neo4j/FalkorDB. Because Graphiti stores embeddings on graph nodes and tracks temporal state natively, there is no "add the graph later" step. It is all the same data, the same database, the same ingest pipeline.
4. **We do not build from scratch.** [[ChampGraph]] already is this stack: Neo4j/FalkorDB, FastAPI, Graphiti, embeddings, OpenRouter. We re-point its schema at the resume domain and reuse ingestion, search, and the D3 viewer. This is the single biggest de-risk.
5. **Apex is not a constraint.** Salesforce calls out to our external service over HTTP. Nothing gets rewritten in Apex. The suite stays in Python and stays portable.
6. **The pitch to the client:** their own procured server. Embeddings computed once and stored on the graph at ingest, never re-embedded per query. Cost and latency stay flat. Data never leaves their box. See [Section 8](#8-cost-model-and-the-dedicated-server-pitch).

---

## 1. The Problem, Decoded

The client searches a **text resume** field holding full text extracted from candidate PDFs and Word documents. They want to search by skills, location, experience, and free keywords, with the match mapping back to the whole resume. The native Salesforce parser is poor.

Two failures were called out specifically:

| # | Failure | What happens | Root cause |
|---|---------|--------------|------------|
| 1 | **Stale experience** | A resume parsed in 2020 had years-of-experience captured as a fixed number. In 2026 it still shows the 2020 value, so it lags by roughly 6 years. | YoE was stored as a frozen scalar at parse time instead of being computed from a date anchor. |
| 2 | **Weak skills capture** | A candidate with 5 major skills only gets 3 captured. Strong candidates get silently dropped when a good project comes in. | Dictionary or regex extraction with low recall. No semantic fallback, so a skill phrased differently is missed entirely. |

The business risk the client cares about: **missing the right candidate** when a good project lands. That is the line to lead with when we talk to them.

---

## 2. Current State (what is already built)

```
Salesforce  ──(extract)──►  Temporary server  ──►  SQL store  ──►  Python search component
                                                                         │
                                                                  phrase / keyword match
                                                                         │
                                                                  pagination (50 / page)
```

- Data was moved out of Salesforce because the search functionality the client wanted does not exist in Salesforce.
- A Python search component runs on our instance and does phrase matching. Searching "java" against roughly 2 lakh records returns all matching records.
- Pagination at 50 resumes per page is in progress.

> [!note] Honest framing
> What exists is fast keyword matching. It does not understand meaning, cannot rank by relevance, and inherits the same bad extraction the client already complained about. The build below makes it intelligent — in one pass.

---

## 3. The Gap (before and after)

| Capability | Keyword search (today) | After this build |
|------------|------------------------|-----------------|
| Find "java" resumes | Yes, all 200k undifferentiated | Yes, ranked by relevance |
| "backend engineer who scaled payments" | Misses anyone who did not write those exact words | Finds them by meaning |
| Rank best-fit first | No | Yes, semantic + graph weighting |
| Recover the dropped 2 of 5 skills | No | Yes — LLM extraction + skill normalization on graph nodes |
| "Who is similar to this strong hire" | No | Yes — graph neighborhood query |
| Adjacent-skill discovery | No | Yes — `SKILL_CO_OCCURS_WITH` edges |
| Correct, current experience | No | Yes — temporal anchor, computed live |
| Company-path matching | No | Yes — `WORKED_AT` edges |
| Pool intelligence ("which skills are thin") | No | Yes — graph aggregation query |

> [!note] Why one phase works
> Because Graphiti stores embeddings directly on Neo4j/FalkorDB nodes and tracks entity state over time, semantic search and graph traversal run against the same data from day one. There is no migration step. Data ingested at the start is already in the graph, ready for relationship queries without re-processing.

---

## 4. What We Build

### 4.1 Fix experience recency (deterministic, no AI)

Stop storing YoE as a frozen number. Store a **date anchor** on the employment edge and compute live.

- **Preferred anchor:** earliest employment start date, extracted at ingest. YoE = `today − earliest_start`, always current.
- **Fallback:** store the captured value + parse date, roll forward by elapsed time.

```python
from datetime import date

def current_years_experience(earliest_role_start: date | None,
                             captured_yoe: float | None,
                             parsed_on: date | None,
                             today: date | None = None) -> float | None:
    today = today or date.today()
    if earliest_role_start:
        return round((today - earliest_role_start).days / 365.25, 1)
    if captured_yoe is not None and parsed_on:
        drift = (today - parsed_on).days / 365.25
        return round(captured_yoe + drift, 1)
    return None
```

Graphiti's bi-temporal layer stores `earliest_role_start` as a valid-from date on the employment edge — not a frozen scalar — so this is structurally sound, not just a calculation hack.

### 4.2 Fix skills capture (LLM extraction at ingest)

Replace the dictionary/regex parser with an LLM extraction pass that runs once per resume at ingest and feeds directly into Graphiti as structured episodes.

Target extraction schema:

```json
{
  "candidate_id": "string",
  "skills": [{"name": "Java", "normalized": "java", "evidence": "Built Spring Boot services", "recency": "current"}],
  "roles": [{"title": "Senior Backend Engineer", "company": "Acme", "start": "2019-04", "end": "present"}],
  "earliest_role_start": "2016-06",
  "locations": ["Bangalore"],
  "education": [{"degree": "B.Tech CSE", "institution": "VIT", "year": 2016}],
  "certifications": ["AWS Solutions Architect"],
  "industries": ["FinTech", "SaaS"],
  "summary": "2-3 sentence neutral synopsis for embedding context"
}
```

Two things make this reliable:

- **Skill normalization.** "JS", "JavaScript", "Node" map to the same canonical `Skill` node. This is what recovers the dropped skills and is the foundation of the graph relationships.
- **Evidence capture.** The snippet that justified each skill gives the recruiter a "why this matched" line — a strong demo moment.

### 4.3 Unified store: Graphiti on Neo4j/FalkorDB

Everything — vectors, graph edges, temporal state, full-text — lives in **one Neo4j/FalkorDB instance** managed by Graphiti. No Postgres, no pgvector, no second store.

| Field group | Storage | Notes |
|-------------|---------|-------|
| Structured fields (skills, roles, dates, location) | Graph nodes + properties | Candidate, Skill, Role, Location, Company, Education nodes |
| Full text resume | Node property + Neo4j full-text index | Keyword safety net via `db.index.fulltext.queryNodes` |
| Embedding vector | Stored on nodes by Graphiti at ingest | Computed once, queried via Graphiti semantic search |
| Date anchor | `earliest_role_start` on employment edge | Powers live YoE; Graphiti bi-temporal tracks role validity |
| Temporal state | Graphiti bi-temporal episodes | Role start/end tracked natively — no separate audit table |
| Skill relationships | `SKILL_CO_OCCURS_WITH` edges | Built automatically as resumes are ingested |
| Candidate similarity | `SIMILAR_TO` edges | Derived from embedding distance at ingest |

### 4.4 Entity and edge model

| Entity | Examples |
|--------|----------|
| Candidate | the person |
| Skill | Java, Spring Boot, Salesforce, Apex |
| Role / Title | Senior Backend Engineer |
| Company | Acme, ex-employers |
| Location | Bangalore |
| Industry | FinTech, SaaS |
| Education | B.Tech CSE, VIT |
| Certification | AWS Solutions Architect |

| Edge | Meaning |
|------|---------|
| `HAS_SKILL` | candidate → skill, with proficiency and recency |
| `WORKED_AS` | candidate → role, with start and end dates |
| `WORKED_AT` | candidate → company, with tenure |
| `STUDIED_AT` | candidate → institution |
| `LOCATED_IN` | candidate → location |
| `SKILL_CO_OCCURS_WITH` | skill → skill, weighted by co-occurrence frequency |
| `SIMILAR_TO` | candidate → candidate, via embedding distance |

### 4.5 Hybrid retrieval

A natural-language query gets parsed once into structured constraints + a semantic vector, then retrieval blends three signals — all against the same Graphiti/Neo4j store.

```
Query: "3 years Salesforce developer in Bangalore, ideally ex-product companies"
          │
          ├─ parse → filters: { role: "Salesforce Developer", min_yoe: 3, location: "Bangalore" }
          │           must/nice skills: ["Salesforce", "Apex"]
          │
          ├─ embed  → query vector
          │
          ▼
  1. Pre-filter by structured constraints (Cypher WHERE: location node, min YoE property, role node)
  2. Vector similarity rank within filtered set (Graphiti semantic search on stored node embeddings)
  3. Blend with Neo4j full-text index score (keyword safety net — no Postgres needed)
  4. Optional cross-encoder rerank on the top 100
          │
          ▼
  Ranked, paginated results + per-candidate "why matched" (from evidence nodes on the graph)
```

Embeddings are stored on graph nodes at ingest. A query only embeds the short query string. The corpus is never re-embedded per search.

### 4.6 Graph queries that are now possible

```cypher
// Candidates with Salesforce + Apex, 3+ years, in Bangalore,
// ranked by adjacent in-demand skills
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
WHERE s.normalized IN ['salesforce','apex']
  AND c.years_experience >= 3
  AND (c)-[:LOCATED_IN]->(:Location {name:'Bangalore'})
WITH c, count(DISTINCT s) AS core
MATCH (c)-[:HAS_SKILL]->(adj:Skill)<-[:SKILL_CO_OCCURS_WITH]-(:Skill {normalized:'salesforce'})
RETURN c, core, count(DISTINCT adj) AS adjacency
ORDER BY core DESC, adjacency DESC
LIMIT 50
```

```cypher
// Find candidates similar to a strong hire
MATCH (anchor:Candidate {candidate_id: $id})-[:SIMILAR_TO]->(c:Candidate)
RETURN c ORDER BY c.similarity_score DESC LIMIT 20
```

```cypher
// Pool intelligence: which skills are thin in the database
MATCH (s:Skill)<-[:HAS_SKILL]-(c:Candidate)
RETURN s.normalized AS skill, count(c) AS candidate_count
ORDER BY candidate_count ASC LIMIT 20
```

### 4.7 API surface

```
POST /search        { query, filters?, page, page_size }  → ranked candidates + total_count
GET  /candidate/{id}                                       → full structured record + evidence
POST /ingest        { resume_text, source_id }             → extract, embed, upsert into graph
GET  /healthz                                              → liveness for the bridge
```

---

## 5. Reuse map from [[ChampGraph]]

We are not starting a new codebase. We are giving an existing one a new domain.

GitHub: `Champ-Deep/Graphiti-knowledge-graph`

| ChampGraph asset | Reuse |
|------------------|-------|
| Neo4j / FalkorDB store | Same store, resume schema — no new DB |
| Graphiti layer | Handles vector storage on nodes, bi-temporal episodes, semantic search — the entire intelligence layer |
| FastAPI REST layer + X-API-Key auth | Same, this is the bridge target |
| Sync / dedup orchestration | Same pattern, resume source instead of email |
| LLM extraction service | Repoint prompt at the resume schema in 4.2 |
| D3.js graph viewer | Showcase visual — almost free |
| Entity / edge schema config | Swap definitions for the resume domain |
| Neo4j full-text index | Already supported — keyword fallback without Postgres |

---

## 6. The Apex Question, Answered

**Short answer: Apex is a thin HTTP client. Nothing gets converted to Apex.**

Salesforce Apex makes outbound HTTP callouts. Salesforce is just one front-end consumer of our service — the same way [[Champmail]] and [[ChampVoice]] consume [[ChampGraph]] over REST today. The intelligence lives entirely in the Python service.

```
Salesforce UI (Lightning component)
        │  user types a query
        ▼
Apex callout  ──HTTP POST──►  Bridge (thin API gateway: auth, rate-limit, logging)
                                      │
                                      ▼
                          Python service (FastAPI)
                                      │
                                      ▼
                          Graphiti on Neo4j / FalkorDB
                          (vectors + graph + temporal + full-text — one store)
                                      │
                                      ▼
            ranked candidate IDs + scores + "why matched"  ──►  back to Apex  ──►  rendered in Salesforce
```

- **The bridge** handles auth (Named Credential keeps secrets out of Apex), rate limiting, request shaping, and logging.
- **Governor limits:** Apex callout ceiling is 120 seconds. Our searches return well under that. Pagination keeps payloads small.
- **Data stays put.** Apex sends only the query text and gets back IDs and short snippets. The resume corpus never round-trips through Salesforce.

---

## 7. Target Architecture (full picture)

```
                         ┌───────────────────────────────────────────────┐
   Salesforce UI ───────►│  BRIDGE  (auth, rate-limit, logging, caching)  │
   (Apex callout)        └───────────────────────┬───────────────────────┘
   Standalone UI ───────►                         │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │   Python service (FastAPI)    │
                                   │   /search /ingest /candidate  │
                                   └───────┬───────────┬───────────┘
                                           │           │
                       query parse + embed │           │ ingest: extract + embed + graph write
                                           ▼           ▼
                          ┌─────────────────────┐   ┌────────────────────────────────────────┐
                          │ On-server AI models  │   │  ONE unified store                      │
                          │ - embeddings (local) │   │  Graphiti on Neo4j / FalkorDB          │
                          │ - extraction LLM     │   │  - embeddings on nodes                 │
                          │ - (OpenRouter opt.)  │   │  - graph edges (skills, roles, etc.)   │
                          └─────────────────────┘   │  - bi-temporal episode tracking        │
                                           ▲         │  - full-text index (keyword fallback)  │
                                           │         └────────────────────────────────────────┘
                                  one-time ingest
                       Temporary server (resume corpus out of Salesforce)
```

One server. One database. No Postgres, no pgvector, no second store.

---

## 8. Cost Model and the Dedicated-Server Pitch

| Bucket | When it happens | How it scales |
|--------|-----------------|---------------|
| **One-time ingest** | Once per resume: LLM extraction + embedding + graph write | Scales with corpus size, paid once. Re-runs only on new or updated resumes. |
| **Marginal per-query** | Per search: embed one short query string, optional light query parse | Tiny and flat. No corpus re-embedding, ever. |

Embeddings are stored at ingest on the graph. A query never touches the corpus models again — it just compares vectors via Graphiti. Cost does not grow with search volume.

> [!warning] Data sovereignty decision
> The call mentioned OpenRouter. That is fine for the MVP and for non-sensitive query parsing. But the pitch to the client is "your data never leaves your server." Calling OpenRouter sends resume text off the box, which undercuts that promise.
>
> **Resolution:** run the embedding model and extraction LLM **on the procured server** (self-hosted open model) for the sensitive resume corpus. Reserve OpenRouter for the short query string or MVP phase only. Resumes are PII — this is the right call.

**The pitch:** their own server, contracted through us, all data resident on it. Resumes extracted and embedded once, on the box, written into the graph. Every search after that is a cheap vector + graph traversal, not a fresh analysis of the whole database.

> [!note] On exact dollar figures
> Do not quote a number until we confirm total resume count and lock the embedding model. The structural model above is what we present now.

---

## 9. MVP Scope (what we showcase in 6 hours to 1 day)

**In scope:**

1. Pull a sample of real resumes from the temporary server (a few thousand, sanitized if PII is a concern).
2. Stand up Graphiti on Neo4j/FalkorDB (reuse ChampGraph instance — no new infra).
3. Ingest pipeline: LLM extraction → Graphiti ingestion (embeddings + temporal edges + graph nodes written automatically).
4. Neo4j full-text index on resume text (keyword fallback).
5. `POST /search` live: Graphiti semantic search + Cypher structured filter + live YoE.
6. Minimal UI or reuse ChampGraph D3 viewer with a search box.
7. **The money shot: side by side with the old keyword search.** Same query, old dump vs. new ranked relevant results.
8. One graph view: pick a skill, show its candidate and company neighborhood — the relationship layer is already there because it was built at ingest, not deferred.

**Explicitly out of scope for the showcase:** Salesforce Apex bridge, self-hosted model hardening, auth. Those are the funded build.

**Success criteria:**

| Proof point | Demonstrates |
|-------------|--------------|
| "java" returns ranked, relevant top results instead of an undifferentiated 200k dump | Relevance ranking |
| A candidate the old parser dropped (3 of 5 skills) now surfaces correctly | Extraction recall fixed |
| Experience shows current years, not the frozen 2020 number | Recency bug retired |
| "backend engineer who scaled payments" returns people who never wrote that phrase | Real semantic search |
| Click a skill, see its candidate and company neighborhood | Graph is live, not a promise |

---

## 10. Build Plan and Sequencing

There is one phase. The MVP steps and the production steps are in the same database from the first ingest.

| Step | Work | Owner |
|------|------|-------|
| 1 | Pull sample resume set from the temporary server | [[Sreedeep Surapaneni]] / dev |
| 2 | Stand up Neo4j/FalkorDB + Graphiti (reuse ChampGraph instance) | dev |
| 3 | Write LLM extraction prompt targeting the resume schema (4.2) | dev |
| 4 | Ingest pipeline: extract → Graphiti episode ingestion (embeddings + temporal edges written automatically) | dev |
| 5 | Create Neo4j full-text index on resume text node property | dev |
| 6 | `POST /search`: Graphiti semantic search + Cypher structured filter + live YoE | dev |
| 7 | Minimal UI or reuse ChampGraph D3 viewer | dev |
| 8 | Side-by-side showcase vs old keyword search | [[Sreedeep Surapaneni]] |
| 9 | Harden extraction recall: normalization dictionary, edge cases in skill parsing | dev |
| 10 | Swap to on-server self-hosted embedding model (sovereignty) | dev |
| 11 | Build out `SKILL_CO_OCCURS_WITH` and `SIMILAR_TO` edge population at ingest | dev |
| 12 | Salesforce Named Credential + Apex callout + Lightning component | SF dev |

Steps 1–8 are the 6-hour-to-1-day showable build. Steps 9–12 are the funded production hardening.

---

## 11. Open Questions and Decisions Needed

1. **Corpus size.** Exact total resume count for ingest cost sizing. ("java" alone returns ~2 lakh, so the full corpus is large.)
2. **Server location.** Where does the procured server sit, and is there a data-residency region requirement?
3. **On-server model vs. API.** Confirm the sovereignty pitch so we commit to self-hosted embeddings before the demo. See the warning in [Section 8](#8-cost-model-and-the-dedicated-server-pitch).
4. **PII handling for the demo.** Sanitize the sample resumes, or is a working subset acceptable as-is?
5. **Relationship to the [[Cirralogix SF-PC Migration — Reference Doc|SF-PC migration]].** Is the resume corpus subject to the same Salesforce access cutoff (first or second week of June/July 2026)? If so, ingest timing is constrained by that deadline.

---

## 12. Appendix: Concrete Skeletons

### 12.1 Search endpoint (FastAPI + Graphiti + Neo4j)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from graphiti_core import Graphiti

app = FastAPI()
graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)

class SearchRequest(BaseModel):
    query: str
    page: int = 1
    page_size: int = 50

@app.post("/search")
async def search(req: SearchRequest):
    parsed = parse_query(req.query)          # LLM -> filters + must/nice skills

    # Graphiti embeds the query string and scores against stored node embeddings.
    # The corpus is never re-embedded.
    semantic_hits = await graphiti.search(req.query, num_results=500)

    # Narrow by structured constraints via Cypher on the same graph
    cypher = """
        MATCH (c:Candidate)-[:LOCATED_IN]->(:Location {name: $location})
        WHERE c.years_experience >= $min_yoe
          AND c.candidate_id IN $candidate_ids
        RETURN c
        ORDER BY c.semantic_score DESC
        SKIP $skip LIMIT $limit
    """
    rows = neo4j_session.run(cypher, {
        "location": parsed.get("location"),
        "min_yoe": parsed.get("min_yoe", 0),
        "candidate_ids": [h.uuid for h in semantic_hits],
        "skip": (req.page - 1) * req.page_size,
        "limit": req.page_size,
    })

    return {
        "page": req.page,
        "page_size": req.page_size,
        "results": [shape(r, parsed) for r in rows],  # shape() adds "why matched" from evidence nodes
    }
```

Keyword fallback via Neo4j full-text index (no Postgres needed):

```cypher
CALL db.index.fulltext.queryNodes("resume_fulltext", $query)
YIELD node, score
RETURN node.candidate_id, score
```

### 12.2 Apex callout (illustrative, uses a Named Credential)

```apex
HttpRequest req = new HttpRequest();
req.setEndpoint('callout:ResumeSearchBridge/search');   // Named Credential holds base URL + auth
req.setMethod('POST');
req.setHeader('Content-Type', 'application/json');
req.setBody(JSON.serialize(new Map<String,Object>{
    'query' => searchText, 'page' => 1, 'page_size' => 50
}));
req.setTimeout(120000);
HttpResponse res = new Http().send(req);
// parse res.getBody() -> ranked candidate IDs + scores -> render in Lightning component
```

The secret never lives in Apex. The Named Credential carries the endpoint and auth.

---

*Related: [[Cirralogix]] · [[Recruit Champ]] · [[ChampGraph]] · [[Champmail]] · [[Cirralogix SF-PC Migration — Reference Doc]] · [[Efforts MOC]] · [[Sreedeep Surapaneni]]*
