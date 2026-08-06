# Notes

- Prefers going deep with concrete, worked examples over abstract explanation.
- Strong existing intuition from data engineering (warehouse tables, feature vectors, distance
  concepts) — use these as anchors/analogies rather than starting from zero.
- Minimal toolset preference — don't introduce new tools/libraries unless there's a clear
  deployment reason. Study broadly, deploy narrowly.
- Domain flavor should show up in examples wherever possible. **Three domains, carried in
  parallel** (widened 2026-08-03 at Yc's request — previously retail only):
  - **Retail** — product catalogs, SKUs, customer search, personalization. Still the primary
    worked example and the one the running numbers (4M SKUs, 500 QPS) are based on.
  - **Life insurance / insurtech** — policy forms and riders, underwriting guidelines, state
    form variations, claims files, agent/underwriter Q&A. Closest to Yc's working context.
    Exercises constraints retail does not: PHI/PII, state-by-state form variation, and
    decision auditability ("which document version informed this underwriting decision?").
  - **Fintech** — transaction and merchant embeddings, KYC/dispute documents, fraud similarity.
    Contributes the *opposite* scale profile: billions of vectors, real-time freshness.
  - **The teaching value is that they give different answers.** Insurance: small corpus, low
    QPS, extreme compliance → pgvector wins overwhelmingly. Fintech transactions: 100M–10B
    vectors at high QPS → the domain where pgvector genuinely breaks. Same capacity model,
    three different conclusions. Use this rather than repeating one domain three ways.
- Runs daily sessions referencing "Week X, Day Y" from the 12-week plan — lessons here should map
  1:1 onto those days so the two systems stay in sync, not duplicate/diverge.
- **Session close-out:** at the end of each lesson session, generate a copy-paste handover prompt
  for the next day's chat (references the next Week/Day topic, instructs the new chat to read
  MISSION.md/RESOURCES.md/NOTES.md + the latest learning-record, and carries forward any open
  threads from that session). Do this automatically without being asked each time.
- **Visuals in lessons:** include simple, eye-catching inline SVG diagrams/graphs in lesson HTML
  where they clarify a concept faster than prose (e.g. embedding-space scatter plots, distance
  comparisons, architecture/flow diagrams). Keep them simple and legible, styled with the vars in
  assets/lesson-style.css, not decorative for its own sake — a diagram should replace or reinforce
  a specific paragraph, not just sit there.

- **Animate memory / cost / comparison content (standing request, added Week 2 Day 1 revision):**
  wherever a lesson deals with **memory footprint, cost, or a side-by-side comparison**, include an
  animation or interactive visual — not just a table. Static tables are fine as the *record*, but
  the concept should land visually first. Working patterns: SMIL-animated SVG (`<animate>`,
  `<animateMotion>`) for process/traversal walkthroughs; growing bar charts for size/cost
  comparisons; small vanilla-JS interactive controls (sliders that recompute and redraw) where the
  reader benefits from plugging in their own numbers. Keep to `assets/lesson-style.css` variables.
- **Depth calibration:** Yc's feedback on the original Week 2 Day 1 was "I was expecting more of
  these" — lessons should err toward *more* worked detail: full parameter tables, explicit
  advantages/disadvantages lists, decision guidance ("choose X when..."), and an applied
  end-to-end build section, not just the conceptual core.

- **Hands-on visualization scripts:** Yc independently built a PCA/MDS 2D projection script (via
  their own coding agent) to visualize embedding distances for the Day 2 sentence example, and
  found it clarifying. Confirmed pattern to repeat going forward — where a concept has a "run this
  on your own data and see it" angle (not just a static inline SVG), offer or build a companion
  script the user runs in their own environment (has API keys/network there, this sandbox doesn't).
  When doing dimensionality-reduction visualizations specifically, flag the PCA-vs-MDS distinction
  (PCA optimizes variance capture, not pairwise-distance preservation; MDS explicitly preserves
  pairwise distances) — easy default assumption to get wrong otherwise.

## Known limitation to flag to user
This container's filesystem resets between sessions. Files written here (`/home/claude/teach/`)
won't persist automatically — they need to be uploaded back into the Claude Project's file store
(or downloaded) so future sessions can pick up the workspace state. Recommend Yc re-upload the
`teach/` folder contents periodically, or we regenerate incrementally from learning-records +
MISSION.md pasted back in at the start of a session.

---

## Week 1, Day 4 — Fine-Tuning Embeddings vs Off-The-Shelf (Deep Dive)
_Completed: 2026-08-03_

### Lesson file
`Claude/week1/day4/0004-fine-tuning-embeddings.html`

### Code files
| File | What it demonstrates |
|------|----------------------|
| `week1/day4/mnrloss_finetune.py` | BGE fine-tuning with MNRLoss + BM25 hard negative mining |
| `week1/day4/hybrid_bm25_rrf.py`  | Hybrid BM25 + dense vector search with Reciprocal Rank Fusion |
| `week1/day4/finetune_roi_calc.py`| ROI calculator: fine-tune vs hybrid — two scenario analysis |

### Sections covered (lesson depth map)
1. **What fine-tuning does to transformer weights** — Q/K/V attention matrices updated via
   contrastive gradients; SVG backprop loop diagram.
2. **Loss functions** — Triplet Loss math (margin m), MNRLoss (in-batch negatives =
   B×(B−1) implicit pairs), Contrastive/Siamese Loss. MNRLoss wins on sample efficiency.
3. **Hard negative mining** — Easy negatives → near-zero gradient → dimensional collapse.
   ANCE approach: BM25 top-K minus known positives = hard negatives. +3–8% nDCG@10.
4. **Hybrid Search physics** — BM25 formula (IDF, TF, k₁=1.2, b=0.75). Why BM25 is perfect
   for exact SKU strings. Semantic gap: zero token overlap → BM25 score=0. Dense fills gap.
5. **RRF fusion** — RRF(d) = 1/(60+rank_a) + 1/(60+rank_b). Scale-invariant (BM25 unbounded
   vs cosine [−1,1]). k=60 default (Robertson et al. 2009).
6. **BEIR benchmark** — nDCG@10 = DCG@10 / IDCG@10. Graded relevance 0/1/2. Four-stage
   decision: baseline → hybrid → reranker → fine-tune only if gap remains >5% nDCG.
7. **Fine-tuning data pipeline** — Minimum 10K pairs. Sources ranked: add-to-cart > click-
   through > human annotation > synthetic GPT-4o. Debiasing requirement for popularity bias.
8. **MLOps burden list** — Catalog drift, query distribution shift, re-embedding pipeline,
   A/B shadow indexing, version registry, index rebuild, latency monitoring. Budget 0.5–1 FTE.
9. **Decision flowchart SVG** — Interview-ready 3-diamond decision tree: tried hybrid? → data
   available? → gap >5% after reranking? → then fine-tune.
10. **5-question quiz** — Tests RRF scale-invariance, MNRLoss in-batch efficiency, dimensional
    collapse, SKU OOV problem, and hybrid-first decision.

### Key numbers to memorize (interview-ready)
- MNRLoss: batch_size 64 → 63 in-batch negatives per query (vs 1 for Triplet Loss)
- Hard negatives: ANCE strategy → +3–8% nDCG@10 over random negatives
- BM25 parameters: k₁ ≈ 1.2–2.0 (TF saturation), b ≈ 0.75 (length norm)
- RRF constant k = 60 (empirically stable)
- Fine-tune minimum dataset: 10,000 (query, positive) pairs
- Fine-tune MLOps cost: 0.5–1 FTE ML Engineer ongoing
- BEIR benchmark: 18 heterogeneous retrieval datasets; primary metric nDCG@10
- "90% of retrieval failures are solved by reranking, not fine-tuning"

### New libraries introduced (with deployment rationale)
- `rank-bm25` — lightweight BM25 implementation for hard-negative mining and hybrid search.
  In production, use Elasticsearch/OpenSearch BM25 index instead.

### Handover prompt for Day 5
> Continuing my AI Data Architect prep — Week 1, Day 5: Embedding Model Justification Framework.
>
> Context & Instructions:
> 1. Please review `Claude/mission.md`, `Claude/dataPrep.md`, and `Claude/NOTES.md` (Day 4 section)
>    for my 14-year Data Engineering background and retail domain target.
> 2. We completed Week 1, Days 1–4:
>    - Day 1: Vector physics & OpenAI embedding models
>    - Day 2: Distance metrics (cosine, dot, Euclidean) + 2-D PCA/MDS visualizations
>    - Day 3: OpenAI vs Open-Source (Bi-Encoders, MRL, privacy levels, PrivateLink)
>    - Day 4: Fine-tuning (MNRLoss, hard negatives, BM25 hybrid, RRF, BEIR nDCG@10, ROI calc)
> 3. Today's Topic (Week 1, Day 5): Embedding Model Justification Framework
>    - How to present a model selection decision to engineering leadership and non-technical
>      stakeholders with numbers, tradeoffs, and clear go/no-go criteria.
>    - Reference lesson artifact: `Claude/week1/day5/0005-embedding-model-justification.html`
>
> Let's dive into Day 5!

---

## Week 2, Day 1 — pgvector Fundamentals: IVFFlat vs HNSW
_Completed: 2026-08-03 · **substantially expanded 2026-08-03** after Yc's feedback ("I was
expecting more of these" + "add animations/images for memory, cost, comparisons")._

### Lesson file
`Claude/week2/day1/0007-pgvector-fundamentals.html`

### Sections covered (after expansion)
1. Physical storage — `vector(n)` as `float4[]` in heap pages. 1536-dim = 6 KB/row.
   **+ interactive memory-footprint explorer** (JS sliders: N, dims, m, instance RAM →
   live bars for heap / IVFFlat / HNSW / HNSW peak-build / your RAM, with a verdict line).
2. Exact kNN vs ANN — brute force O(N × d); 4M × 1536 = 6.14B float mults/query. Exact search
   framed as **the ground truth for recall measurement**, not just the slow fallback.
   **+ animated recall-vs-latency dial** (recall saturates, latency keeps rising).
3. IVFFlat — how it works, **animated query walkthrough** (query arrives → centroid distances →
   probed cell lights up → sequential scan flashes), full parameter table, pros/cons columns.
4. HNSW — how it works, **animated layer-descent traversal** (marker hops L2→L1→L0), full
   parameter table, pros/cons columns.
5. Comparison — **animated 4-row bar chart** of the dimensions that actually differ, plus a
   13-row comparison table.
6. **Which should you choose** — "choose HNSW when / choose IVFFlat when" + rule of thumb.
7. Practical pgvector queries + the pre-filter dilemma (now updated with pgvector 0.8
   iterative scans, `SET hnsw.iterative_scan = relaxed_order`).
8. **NEW — Applied: a product-specific RAG on pgvector.** Pipeline SVG with animated pulse,
   schema + HNSW index DDL, retrieval query with `product_id` filter, the "always filter by
   product" requirement, and the five things that matter more than index choice (chunking,
   metadata, retrieval count 3–8, reranking 10–20→3–5, prompt grounding).
9. **NEW — Benchmark on your own data**: latency P50/P99, recall vs exact, build time, index
   size, insert/update performance.
10. Quiz expanded from 4 to 7 questions.

### The correction embedded in the memory explorer
Common wisdom "IVFFlat uses much less memory than HNSW" is **barely true at 1536 dims**.
Both store a full copy of every vector — "Flat" means uncompressed, and HNSW needs the vector
at each node to compute distance. HNSW's only extra is the neighbour list (2m × 8 B).

Verified with the same model the calculator uses (4M rows, m=16):

| dims | IVFFlat | HNSW | HNSW larger by |
|---|---|---|---|
| 128 | 2.3 GB | 3.5 GB | **50%** |
| 384 | 6.9 GB | 8.1 GB | 17% |
| 768 | 13.7 GB | 14.9 GB | 8.5% |
| 1536 | 27.4 GB | 28.6 GB | **4.3%** |
| 3072 | 54.7 GB | 55.9 GB | 2.1% |

**The real HNSW memory cost is peak BUILD memory (~2× final index), not steady state.**
At 4M × 1536 that's ~57 GB — which is what actually fails on a 64 GB box.

### Key numbers
- 4M × 1536-dim: heap 24.7 GB · IVFFlat 27.4 GB · HNSW 28.6 GB · HNSW build peak ~57 GB
- IVFFlat `lists` ≈ sqrt(N); 4M rows → ~2000. probes=1 ≈ 85% recall, probes=20 ≈ 95%
- HNSW defaults: m=16, ef_construction=64, ef_search=40 (must be ≥ k)
- IVFFlat cold-start: cannot build on an empty table (k-means needs rows)
- Product RAG: HNSW with defaults; retrieve 10–20, rerank to 3–5; **always filter by product_id**

### Animations also retrofitted to other days (same standing request)
- **Day 2 §9** — animated stacked cost bars (infra / sync-pipeline build / ongoing ops) for all
  four options at 4M vectors. Widths verified against `vectordb_tco_calc.py` output. Makes the
  argument visually: pgvector's advantage is the *missing* sync-build segment, not the infra bar.
- **Day 3 §2** — the RAM-cliff chart now draws itself, with a marker that rides the flat green
  section and turns red as it falls off the edge.
- **Day 3 §7** — animated mitigation-ladder bars showing bytes/row down each rung, with the
  fixed 280 B floor (2m links + header) drawn as a line to show visually why 2×3 ≠ 6×.

---

## Week 2, Day 2 — Pinecone / Weaviate / Milvus vs pgvector
_Completed: 2026-08-03_

### Lesson file
`Claude/week2/day2/0008-pinecone-weaviate-milvus.html`

### Code files
| File | What it demonstrates |
|------|----------------------|
| `week2/day2/dual_write_drift_sim.py`  | Simulates CDC-synced external vector store — drift, DLQ, orphans |
| `week2/day2/vectordb_tco_calc.py`     | 4-way TCO across scale tiers incl. sync-pipeline engineering cost |

Both stdlib-only, deterministic seed, verified running against the project venv.

### Sections covered (lesson depth map)
1. **The reframe** — not "which DB has better recall" but "where does vector state live
   relative to the source of truth?" Co-located (pgvector) vs disaggregated (everyone else).
2. **The dual-write problem** — pgvector's structural advantage. Stale vectors and orphaned
   vectors as the two failure modes; transactional outbox → CDC → reconciliation as the fix
   you must build. SVG: one commit vs two systems.
3. **Pinecone** — serverless storage/compute separation, namespaces as the multi-tenancy
   primitive, no index knobs. Costs: eventual consistency, no joins, duplicated metadata,
   closed source (no exit path).
4. **Weaviate** — native hybrid (`alpha` + rankedFusion/relativeScoreFusion), tenant shards
   with ACTIVE/INACTIVE/OFFLOADED lifecycle. Vectorizer modules flagged as an anti-pattern
   for a data-platform team (moves embedding generation into an uncontrolled request path).
5. **Milvus** — disaggregated architecture SVG (access/coordinator/worker/storage), log-as-data,
   isolated index nodes, 4 explicit consistency levels, quantization + DiskANN + GPU index zoo.
6. **Qdrant sidebar** — filterable HNSW; plus the correction that pgvector 0.8 iterative index
   scans largely close Day 1's pre-filter gap.
7. **Capability matrix** — 11 rows across all four systems.
8. **Multi-tenancy SVG** — four models ranked; the key argument against shared-index+filter is
   *recall degradation for small tenants*, not security.
9. **Cost/ops model** + the four migration triggers with thresholds.

### Key points to memorize (interview-ready)
- The dual-write problem is the strongest pro-pgvector argument and the one GenAI engineers miss
- "Managed covers *their* index, not *your* correctness" — the sync pipeline is yours either way
- Shared HNSW + tenant_id filter → recall becomes a function of tenant size (silent failure)
- Migration triggers with numbers: ~50M+ vectors (volume) · hundreds of skewed tenants (tenancy) ·
  ingest saturating query capacity (write velocity) · no Postgres ops capability (team shape)
- Escape hatches to try before migrating: pgvectorscale (StreamingDiskANN), ParadeDB pg_search
  (real BM25 in Postgres), AlloyDB ScaNN, Citus
- pgvector has partially closed the quantization gap: halfvec/binary/sparsevec in 0.7,
  iterative index scans in 0.8 — asserting "pgvector has no quantization" dates you

### Simulator findings worth quoting
- 4M SKU catalog, 24h, realistic churn → ~221k SKU-minutes of drift, ~158 SKUs drifted at any
  instant, ~17k customer searches/day touching a stale result
- Reconciliation sweeps rescue *transient* loss only; poison writes (schema/dimension mismatch,
  oversized metadata, unknown namespace) reproduce on every retry and still need a human
- TCO crossover: pgvector is cheapest all-in through ~20M vectors; at 100M it stops being lowest
- Peak-vs-average QPS is the most common modelling error — provisioned options are sized for
  peak, consumption billing follows average (~4× difference on a diurnal retail profile)

### Handover prompt for Day 3 (superseded — Week 2 completed in full, see below)
> Continuing my AI Data Architect prep — Week 2, Day 3: Scale Tradeoffs — when does pgvector
> start to struggle?
>
> Context & Instructions:
> 1. Please review `Claude/mission.md`, `Claude/dataPrep.md`, and `Claude/NOTES.md`
>    (Week 2, Day 1 and Day 2 sections) for my 14-year Data Engineering background and
>    retail domain target.
> 2. Week 1 covered embeddings end-to-end (vector physics → distance metrics → OpenAI vs
>    open-source → fine-tuning/hybrid/RRF → justification framework → 1-pager checkpoint).
>    Week 2 so far:
>    - Day 1: pgvector fundamentals — storage layout, IVFFlat vs HNSW, pre-filter dilemma
>    - Day 2: Pinecone/Weaviate/Milvus — dual-write problem, native hybrid, multi-tenancy
>      models, capability matrix, migration triggers
> 3. Today's Topic (Week 2, Day 3): Scale tradeoffs — at what data volume and QPS does
>    pgvector actually start to struggle vs a purpose-built vector DB? I want concrete
>    numbers: RAM/index-fit math, HNSW build times, vacuum/bloat behaviour, connection and
>    replica limits, and where each one bites first.
>    - Reference lesson artifact: `Claude/week2/day3/0009-pgvector-scale-limits.html`
> 4. Carry forward: Day 2 named four migration triggers with rough thresholds. Day 3 should
>    pressure-test the *volume* one specifically and turn it into a defensible number.
>
> Let's dive into Day 3!

---

## Week 2, Day 3 — Scale Tradeoffs: Where pgvector Actually Breaks
_Completed: 2026-08-03_

### Lesson file
`Claude/week2/day3/0009-pgvector-scale-limits.html`

### Code files
| File | What it computes |
|------|------------------|
| `week2/day3/hnsw_capacity_planner.py` | Wall-by-wall capacity analysis; tells you which wall you hit first |
| `week2/day3/mitigation_ladder.py`     | Cumulative effect of each mitigation rung + runway in years |

### The correction that matters (supersedes Day 1)
Day 1's "HNSW adds 30–60% overhead" is **wrong for capacity planning**. pgvector's HNSW index
stores a **full copy of every vector** — traversal needs the vector at each node to compute
distance. Real figure is **105–115% of the heap** at 1536 dims.

    hnsw_per_row = dims×4 + (2×m)×8 B + ~24 B, over a ~0.9 fill factor
                 = 6,144 + 256 + 24 → ~7,138 B/row → 4M rows = 28.6 GB index

Consequence: the vector copy is ~96% of the index, so **every effective lever shrinks the
vector, not the graph**.

### The five walls (in the order you hit them)
1. **RAM** — the cliff. HNSW traversal is *serially-dependent pointer chasing* at queue depth 1,
   so high-IOPS SSDs don't help; you pay per-read latency once per hop. 2 ms cached → 10 ms
   local NVMe → 50–100 ms network SSD. Leading indicator: the HNSW index's own cache-hit ratio
   in `pg_statio_user_indexes`, NOT the database aggregate.
2. **CPU** — ~200–400 QPS/core at 1536 dims, shared with OLTP. At 500 QPS you consume 1.5–2.5
   cores continuously. Under a spike **checkout fails before search does**.
3. **Writes** — HNSW inserts mutate the graph; deletes leave tombstones that remain as traversal
   waypoints. VACUUM does not restore graph quality. REINDEX CONCURRENTLY needs 2× index disk.
4. **Replicas** — replication scales QPS but **not corpus size**; each replica holds a full copy.
   This is the structural reason pgvector has a volume ceiling and Milvus doesn't.
5. **Connections** — Little's Law. The cliff compounds: 500 QPS × 2 ms = 1 connection, but
   × 80 ms = 40. PgBouncer transaction mode before you need it.

### Mitigation ladder (measured, not multiplied)
| Lever | Real gain | Recall cost |
|---|---|---|
| halfvec (fp16) | 1.9× | ~0.3% |
| MRL 1536→512 | 2.8× | ~2% |
| **Both combined** | **4.9×** | ~2.3% |
| Binary + exact rerank | 13.6× | recoverable |

**The 6× trap:** 2 × 3 = 6 overstates it. Only the vector shrinks — the neighbour list
(2m × 8 B = 256 B) and element header are fixed costs. Per row 6,424 B → 1,304 B = 4.9×.
Interviewers ask you to derive this; don't multiply in your head.

### Defensible numbers
- ~7 GB of index per 1M vectors at 1536 dims
- <10M trivial · 25–50M viable with the ladder · >100M migration territory
- Expect the **CPU** wall before the RAM wall if search shares the OLTP instance →
  first move is a dedicated read replica, not a new database

---

## Week 2, Day 4 — Hybrid Search in Production
_Completed: 2026-08-03_

### Lesson file
`Claude/week2/day4/0010-hybrid-search-production.html`

### Code files
| File | What it demonstrates |
|------|----------------------|
| `week2/day4/query_class_router.py`    | Regex/heuristic classifier → per-class RRF weights |
| `week2/day4/hybrid_eval_by_class.py`  | The evaluation trap, computed numerically |

_Distinct from Week 1 Day 4 (BM25 math, RRF theory) — this is the production build._

### Sections covered
1. **Two gaps, not one** — semantic gap (BM25 scores 0 on zero overlap) AND lexical gap
   (dense can't do exact match). The lexical gap is a **confident precision failure**: an
   embedding is a smooth function, so SKU-88214 embeds near SKU-88213 and is returned at
   rank 1 with a high score. Nothing in monitoring looks wrong.
2. **Retail query taxonomy** — exact identifier / brand+model / attribute-constrained /
   natural language / misspelled, each wanting different treatment.
3. **Lexical side in Postgres** — generated `tsvector` column (the Day 2 advantage applied to
   search: lexical index maintained by the *same commit*), `setweight` for field importance,
   separate `sku_norm` btree for identifiers, pg_trgm for fuzzy.
4. **Full hybrid SQL** — two CTEs + weighted RRF, over-fetch 100 per arm.
5. **Fusion** — RRF vs normalised score blend; bypass fusion entirely for exact identifiers.
6. **The evaluation trap** — the centerpiece.
7. **Operating two indexes** — GIN fastupdate, planner verification, failure isolation.

### Gotchas worth memorizing
- `to_tsvector(text)` (1-arg) is only STABLE → **rejected in a generated column**. Use the
  2-arg form `to_tsvector('english', ...)` which is IMMUTABLE. Trips up everyone once.
- **`ts_rank_cd` is not BM25 — it has no IDF term at all.** No corpus statistics, so a rare
  discriminative model number is weighted the same as a common word. Exactly backwards for
  retail. Softened by RRF (rank-only), fixed properly by ParadeDB `pg_search`.
- 'english' config lowercases and splits `SKU-88213` into `sku` + `88213` → never rely on FTS
  for exact identifier lookup.

### The evaluation trap (numbers from the script)
| Class | Traffic | Conv | Dense | Blanket | Δ |
|---|---|---|---|---|---|
| NATURAL_LANGUAGE | 25% | 1.0× | 0.58 | 0.71 | +0.13 |
| ATTRIBUTE_CONSTRAINED | 30% | 1.4× | 0.62 | 0.70 | +0.08 |
| MISSPELLED | 10% | 0.8× | 0.44 | 0.63 | +0.19 |
| BRAND_MODEL | 27% | 2.1× | 0.73 | 0.70 | −0.03 |
| **EXACT_IDENTIFIER** | **8%** | **4.3×** | **0.94** | **0.71** | **−0.23** |

- traffic-weighted: 0.647 → 0.696 (**+0.049, reported and shipped**)
- revenue-weighted: 0.709 → 0.700 (**−0.009, the number that matters**)
- routed (identifiers short-circuit to btree): revenue-weighted **+0.082**, every class improves

**Rule: never evaluate retrieval on an aggregate alone. Segment by query class and weight by
business value per class, not traffic share.**

---

## Week 2, Day 5 — Operational Simplicity
_Completed: 2026-08-03_

### Lesson file
`Claude/week2/day5/0011-operational-simplicity.html`

### Code files
| File | What it's for |
|------|---------------|
| `week2/day5/pgvector_ops_healthcheck.sql` | 9-section read-only health check for a real pgvector DB |
| `week2/day5/recall_monitor.py`            | Golden-set recall job; demo shows real decay, latency flat |

### The centerpiece: recall is unmonitored by default
**Your dashboards measure whether the query ran. Nothing measures whether it returned the
right rows.** Latency, throughput, error rate and cache-hit ratio can all be green while
approximation quality decays from graph churn, stale centroids, distribution drift, or
someone lowering `ef_search` to silence a latency alert.

Because latency and recall trade against each other, **an unmonitored recall metric means
every latency optimisation is an unmeasured quality regression.**

Measured in the demo (real approximate index, real exact-search ground truth, 6 months of
churn at 12% delete / 15% insert per month):

    recall@10   0.990 → 0.867   (SLO 0.90 breached at month 4)
    work/query  417 → 444 distance computations (+6.4% — trips no alert ever configured)

**Why it's a tractable job, not a research project: exact search IS the ground truth.**
`SET LOCAL enable_indexscan = off` → true top-10; compare to the ANN top-10. No human labels.
Expensive, which is why it belongs at 3am on a read replica, not in the request path.
Alert on the mean; **page per query class** (Day 4's lesson).

### Also covered
- What you inherit free: backup/PITR, replication, RBAC, pooling, migrations, DR runbooks —
  but the two that actually decide it are **compliance sign-off** (a new store restarts
  security review and data-residency assessment) and **on-call competence** (at 3am the
  person paged can already read a query plan).
- What pgvector adds that is NOT free: memory sizing, `maintenance_work_mem` for builds
  (too small → build spills and takes ~5× longer), REINDEX cadence, extension/major-version
  upgrade coupling, workload isolation.
- **Honest counter-case — blast radius.** Search and OLTP share a failure domain; a 53 GB
  vector table slows PITR restore for the *whole* database. Mitigation: dedicated read
  replica or a separate instance fed by logical replication. Naming this before an
  interviewer does shows you've thought past the slogan.
- Embedding-model migration (annual in practice) is a schema migration in Postgres —
  ADD COLUMN, backfill, flip, drop — all transactional and rollback-able, both versions
  coexisting for A/B. Across a dual-write boundary it's a sync project.
- Managed vector DBs don't monitor your recall either — and may offer no exact-search mode,
  so you can't compute ground truth without exporting every vector.

---

## Week 2, Day 6-7 — CHECKPOINT: pgvector Justification & Scale Boundary
_Completed: 2026-08-03_

### Deliverable
`Claude/week2/day6-7/0012-checkpoint-pgvector-justification.html` — ADR-002, portfolio-ready,
same format as Week 1's ADR-001.

Contains: context (Personal Agent + retail reference architecture) · 4 options evaluated ·
decision + 6-point rationale · capacity model with the per-row formula · the four migration
triggers with numeric thresholds · escape-hatch ladder · consequences accepted and follow-on
work required · **three interview drill answers** (why pgvector / when does it break / what
would you monitor) · Week 2 decisions log · 6-question mastery quiz.

### The framing that makes the answer strong
Most candidates answer "why pgvector" with *preferences* (simplicity, familiarity, cost).
This ADR answers with a **capacity model and named triggers**. "pgvector is correct until
25M vectors, then lever 3 buys us to 100M, and past that the trigger is X" is the difference
between a senior engineer and an architect.

### Decisions log added to `dataPrep.md` running list
- Why pgvector over Pinecone/Weaviate — co-location makes freshness a transaction, not a pipeline
- Why HNSW over IVFFlat — recall/latency + incremental inserts; IVFFlat centroids go stale
- Why tsvector generated column + weighted RRF — lexical index maintained by the same commit
- Why class-routed hybrid over a global alpha — a global alpha regresses the highest-converting segment
- Why a dedicated read replica — removes OLTP contention without changing databases
- Why nightly golden-set recall@10 — the only signal that detects silent retrieval decay

---

## Handover prompt for Week 3, Day 1

> Continuing my AI Data Architect prep — Week 3, Day 1: Chunking Strategies.
>
> Context & Instructions:
> 1. Please review `Claude/mission.md`, `Claude/dataPrep.md`, and `Claude/NOTES.md`
>    (all of Week 2) for my 14-year Data Engineering background and retail domain target.
> 2. Completed so far:
>    - **Week 1 (Embeddings):** vector physics → distance metrics → OpenAI vs open-source →
>      fine-tuning/MNRLoss/hard negatives/BM25+RRF/BEIR → justification framework → ADR-001.
>    - **Week 2 (Vector Databases):** pgvector internals (IVFFlat vs HNSW) → Pinecone/Weaviate/
>      Milvus and the dual-write problem → the five scale walls and the mitigation ladder →
>      production hybrid search with query-class routing → operational simplicity and
>      golden-set recall monitoring → ADR-002 checkpoint.
> 3. Today's Topic (Week 3, Day 1): Chunking strategies — fixed-size vs semantic vs recursive,
>    and how chunk size affects retrieval quality. I want the same treatment as Week 2:
>    concrete numbers, a retail/agent worked example, and a runnable companion script.
>    - Reference lesson artifact: `Claude/week3/day1/0013-chunking-strategies.html`
> 4. Carry forward:
>    - Week 2 established that **recall must be measured, not assumed** — chunking decisions
>      should be evaluated with the same golden-set discipline.
>    - Week 2 Day 4 established that **aggregate metrics hide per-class regressions** — apply
>      that lens to chunk-size evaluation too.
>    - My Personal Agent (~10k chunks) is the running example; the 4M-SKU retail platform is
>      the reference architecture for scale questions.
>
> Let's dive into Week 3, Day 1!

---

## Week 2 — Domain expansion to life insurance / insurtech + fintech
_Applied: 2026-08-03, at Yc's request. `mission.md` updated to match._

Retail remains the primary worked example and the source of the running numbers. Life insurance
and fintech now run alongside it in every Week 2 lesson via a three-column `.domains` component.

### The organising idea
**Same capacity model, three different answers.** That is the teaching value — not repeating one
domain three ways. An architect who can say "for a 200k-chunk underwriting corpus it isn't close,
but for 2B transaction embeddings I'd be sharded, and here's the row count where I'd switch" has
a model; one who says "pgvector is great" has a preference.

| | corpus | QPS | index | verdict |
|---|---|---|---|---|
| Retail | 4M × 1536 | 500 | 28.6 GB | pgvector; **CPU** is the first wall → read replica |
| Insurance | 200k × 1536 | 20 | 1.4 GB | pgvector, not close; **no wall reachable** (35× growth needed) |
| Fintech docs | 150k × 1536 | 30 | 1.1 GB | looks like insurance, not like fintech-txn |
| Fintech txns | 2B × 768 | 4,000 | **7.4 TB** (needs 13.4 TB) | **pgvector breaks** — 26× a 512 GB box; full ladder still ~1.5 TB → shard |

### Domain-specific content added, by day
- **Day 1** — three-domain scenario block; memory-explorer slider max raised to 2.5B so the
  fintech case is reachable. **Insurance RAG variant with bitemporal filtering**: a claim may need
  adjudicating against the form version *in force at issue*, possibly a decade old → keep every
  version, filter on an effective-date range, never delete. Pleasant side effect: an append-only
  corpus has **no graph decay from deletions**, which is the Day 5 problem solved by the data
  model. Also: "fintech RAG" hides two different workloads (docs vs transactions).
- **Day 2** — drift cost by domain; **PHI/PII section**: an embedding derived from PHI is still
  that data (embedding inversion), so the vector store enters compliance scope — vendor review,
  DPA, BAA, data residency, right-to-erasure evidencing, and RLS that doesn't extend into an
  external index. Held honestly: this argues for *data locality*, and self-hosted Weaviate/Milvus
  in-VPC also satisfies residency — what pgvector uniquely keeps is deletion/RLS/audit as **one
  system, one transaction**. Insurtech multi-tenancy = the 400-storefront shape with carrier trade
  secrets and contractual isolation.
- **Day 3** — three-domain wall comparison. Includes the trap: "no wall is reachable" is right but
  sounds like you skipped scale — state it as a *bound* ("35× growth before RAM is a conversation,
  so the engineering belongs on retrieval correctness").
- **Day 4** — identifier grammars per domain. **The medical-code case is the sharpest lexical-gap
  example available**: ICD-10 `E11.9` vs `E11.29` embed almost identically but can mean
  accept-at-standard vs decline. Must be exact lookup, never fusion. Fintech merchant descriptors
  (`AMZN MKTP US*2H4TR`) are the canonical hybrid problem.
- **Day 5** — recall decay cost by domain, and why the regulated cases make the golden-set job
  non-optional: **no downstream signal catches it in time** (loss ratios move quarters later).
  Plus **reproducing a past retrieval** for audit — log doc IDs *and versions*, embedding model
  version, index parameters, the query embedding, and the effective-date filter.
- **Day 6-7** — ADR §7 Domain Applicability + expanded context; explicit warning against
  over-generalising ("pgvector is the right default" is *not* what the ADR concludes).

### Script changes
- `hnsw_capacity_planner.py --domain {retail|insurance|fintech-txn|fintech-docs}` — presets with
  per-domain notes. Fixed a wrong recommendation surfaced by the fintech preset: it used to
  suggest the mitigation ladder unconditionally; it now checks whether the *best* rung (binary +
  exact rerank) actually closes the gap and says "LADDER EXHAUSTED → shard" when it can't.
- `query_class_router.py --domain {retail|insurance|fintech}` — per-domain identifier grammars,
  a new `MEDICAL_CODE` class (short-circuits, fails loudly on unknown codes rather than returning
  a near match), and a `MERCHANT_STRING` class. The merchant case was initially misclassified as
  NATURAL_LANGUAGE with dense-dominant weights — actively wrong for merchant normalisation, since
  the brand token is the high-signal part and the acquirer prefix is noise. Now 0.65/0.35 lexical-
  dominant with a pg_trgm arm.

### Recurring wedge to reuse in interviews
Regulated-domain RAG corpora are **versioned dimensions with validity windows**, not folders of
documents to embed. Bitemporal retrieval, append-only audit logs, and lineage sufficient to prove
what the system returned in March are SCD problems — 14 years of exactly this. Most GenAI
engineers have never been asked to reproduce a retrieval from a past date; in insurance and
fintech that question is routine.

---

## Week 2, Day 2 — Added §11–13 (other RDBMS, Postgres RAM limits, distributed vectors)
_Added: 2026-08-04, answering three questions from Yc._

### §11 The Wider Field — which other RDBMS have vectors
13-row table: Oracle 23ai, SQL Server 2025, MariaDB 11.7+, MySQL 9.x, SingleStore, ClickHouse,
DuckDB, Snowflake/BigQuery, Cassandra 5.0, MongoDB Atlas, Elasticsearch/OpenSearch, Redis.

**The strategic read, not the feature list:** every major RDBMS shipped a native vector type
between 2023–2025 → the industry decided vector search is a **datatype, not a product category**.
That is exactly the bet co-location makes, and it's the answer to "isn't pgvector a toy?"

Traps and honest concessions worth memorizing:
- **MySQL 9.x community has the VECTOR type but NO ANN index** — searches are brute force.
  Indexed vector search is HeatWave (paid) only.
- **Oracle 23ai has a dedicated vector memory pool** (`VECTOR_MEMORY_SIZE`) instead of competing
  with the buffer cache — genuinely a better design than pgvector's.
- **SQL Server 2025 ships DiskANN natively**; pgvector needs pgvectorscale for that.
- Naming where pgvector is *behind* is what makes the rest of the argument credible.

### §12 Postgres memory limits — four ceilings, only one binds
1. **Postgres itself: effectively none.** `shared_buffers` max ≈ 16 TB. Not the constraint.
2. **Don't raise `shared_buffers` to fix vector caching.** Postgres uses *buffered* I/O, so pages
   sit in shared_buffers AND the OS page cache → double buffering wastes the memory. **What
   matters is total machine RAM**; an OS-cached page is just as fast. Size the instance, leave
   shared_buffers conventional. (Good interview answer — most people reach for the wrong lever.)
3. **Per-connection memory multiplies** — work_mem per plan node per connection;
   maintenance_work_mem too small → HNSW build spills to disk, ~5× slower.
4. **Instance ceiling** ~1–4 TB managed / tens of TB bare metal. Against ~7 GB per 1M vectors at
   1536 dims, a 1 TB box holds ~80M vectors before the ladder, several hundred million after.

**The one that actually bites — pgvector's index dimension limit (NOT a memory limit):**
an index entry must fit one 8 kB page.

| type | column max | **index max** |
|---|---|---|
| `vector` (fp32) | 16,000 | **2,000** (2,000 × 4 B = 8,000 B) |
| `halfvec` (fp16) | 16,000 | **4,000** (4,000 × 2 B = 8,000 B) |
| `bit` | — | **64,000** |

→ **`text-embedding-3-large` (3,072 dims) cannot be HNSW-indexed as `vector(3072)`.** Column works,
inserts work, `CREATE INDEX` fails — so teams hit it *after* embedding a whole corpus. Fix:
`halfvec(3072)` or MRL-truncate to ≤2,000. Reframes halfvec: at 3,072 dims it isn't an
optimisation, **it's the only way to have an index**. Also cross-referenced into Day 1 next to the
memory explorer (which offers a 3072 setting).

### §13 Distributed vector search — how sharding actually works
**The core difficulty: nearest-neighbour search has no partition key.** `WHERE customer_id = 42`
routes to one shard; "find the 10 most similar" can be answered by any shard.

Three strategies:
- **A. Random shard + scatter-gather** (Milvus, Elasticsearch, Vespa). Recall preserved, volume
  scales — but **QPS does NOT scale with shards** (every query hits every shard), and you wait for
  the slowest shard every time.
  - **Tail amplification** (Dean & Barroso, "The Tail at Scale"): each query samples every shard's
    distribution and takes the MAX. 10 shards at P99=10 ms → ~10% chance some shard is in its tail,
    so query-level P90 looks like shard-level P99. **Sharding adds a latency problem you didn't
    have.** Animated SVG shows one straggler dominating five fast shards.
- **B. Semantic sharding** (route by vector-space region, SPANN-style). QPS scales, but recall
  suffers at region boundaries and rebalancing means moving vectors between nodes.
- **C. Shard on a business key — the one to want.** `storefront_id` / `carrier_id` / `tenant_id`.
  Every query single-shard → no scatter, no tail amplification, no recall loss, QPS scales
  linearly, and each shard's index stays resident (defuses the RAM cliff too).
  **A tenancy model IS a sharding strategy** — connects Day 2 §8 to Day 3's volume wall.

Escalation ladder: declarative partitioning (still plain Postgres) → Citus (shards pgvector tables,
same SQL) → AlloyDB/Aurora → Yugabyte/CockroachDB → SingleStore/TiDB → Milvus/Vespa.

Quiz grew 6 → 11 questions covering the dimension limit, shared_buffers, tail amplification,
business-key sharding, and the "pgvector is a toy" pushback.

---

## Week 2, Day 3 — Added §8 "Building the Index Outside Postgres"
_Added: 2026-08-04, answering Yc's question about external C++ index building._

### The two-part answer
**Half 1 — you CANNOT import a prebuilt index into pgvector.** A pgvector index is a Postgres
*access method*, not a file: relation page format, WAL-logged, participates in MVCC visibility,
tied to physical tuple IDs. Postgres exposes no "load a prebuilt index" API for *any* index type.
So no path exists where you build with hnswlib in C++ and copy bytes in.
- In-Postgres workaround: **shadow table + atomic rename swap** (`CREATE TABLE products_v2` →
  build index → `BEGIN; ALTER TABLE ... RENAME; COMMIT`). Versioned, instantly rollback-able,
  cutover in milliseconds. Still burns build CPU on the host; costs 2× storage in transition.

**Half 2 — you CAN move the index out entirely, and it's a real architecture.** Treat the index
as a **derived build artifact**: Postgres = source of truth for vectors, ANN index = versioned
file built by a batch job and served by something else. Same shape as a nightly aggregate table
or offline→online feature-store materialisation.

C/C++ libraries: **FAISS** (Meta, GPU), **hnswlib** (header-only, by the HNSW authors),
**usearch** (mmap-able, used in ClickHouse), **DiskANN/Vamana** (SSD-resident), **ScaNN**
(Google, also what AlloyDB uses), **Annoy** (Spotify, mmap'd immutable — built for this pattern).

### Base + delta + tombstones — how you get freshness from an immutable artifact
- **Base** — large, immutable, rebuilt on a schedule (~99% of vectors)
- **Delta** — small, in-memory, rows changed since the base was cut
- **Tombstones** — IDs deleted/updated since the base; filtered from results
- Query = base + delta → merge → drop tombstones → top-k

**This is an LSM-tree with vectors in the leaves.** Yc has reasoned about write amplification and
compaction cadence before — same tradeoff. The failure mode to name: *the delta grows until
querying it costs more than the base did*, so **compaction cadence is set by delta size, not by
the clock.**

### The honest framing (important — don't oversell this)
Moving the index out is **not** a way to dodge Day 2's dual-write problem — it's a way to *accept*
it in exchange for build isolation and index freedom. The difference from adopting Pinecone is
real but narrow: the artifact is **derived and disposable**, so failure costs a rebuild rather
than data loss, and Postgres stays the single authoritative copy. That is a genuinely weaker
failure mode; it is *not* "no sync pipeline."

Costs incurred: dual-write/freshness lag · no SQL `WHERE` in the index (pre-filter dilemma returns
worse) · no MVCC · tombstones maintained separately · you own a serving service · two round trips
(index for IDs, Postgres to hydrate).

**Design rule that falls out:** use the external index for **RANKING only**. Anything the user
must see immediately (price, stock, balance, policy status) comes from Postgres at hydration time,
never from index metadata.

### Middle ground worth knowing
| approach | build runs | query runs |
|---|---|---|
| plain pgvector | in PG | in PG |
| shadow table + rename swap | in PG, your schedule | in PG |
| `pgvectorscale` | in PG (Rust, StreamingDiskANN) | in PG, SSD-designed → Wall #1 softens |
| **Lantern** | externally via CLI, then imported | in PG — closest to "build outside, serve inside" |
| external artifact (FAISS/usearch) | batch/GPU box | own service; PG hydrates |

### New companion script
`week2/day3/external_index_tradeoff.py` (+ `--domain retail|insurance|fintech-txn`) — sweeps
rebuild cadence vs query cost vs freshness lag, finds the cheapest cadence within a query-inflation
budget. Concrete results:
- **retail** (4M, 3%/day churn): nightly rebuild inflates queries **7.75×**; 15-min rebuilds cost
  **1.07×** and 0.22 of a build machine → 15-min lag is the honest price.
- **insurance** (200k, 0.1%/day): trivial at any cadence — the reason to use an external artifact
  here is **reproducibility for audit**, not performance.
- **fintech-txn** (2B, 2%/day): **no cadence works** — even 15-min rebuilds leave a 417k-vector
  delta inflating queries 24×, and sustaining it needs 8.9 build machines. Correct conclusion:
  index the delta properly, or shard. Good demonstration that the model says "don't" when it should.

### Per-domain verdict on whether to bother
retail 4M → over-engineering; shadow-table swap solves Wall #3 without a new service.
insurance 200k → not worth it on performance, but immutability is attractive for audit.
fintech 2B → not optional; no in-Postgres lever reaches and PQ/GPU builds are the only affordable way.

Day 3 quiz grew 6 → 9 questions. Sections renumbered: §8 inserted, old 8→9, 9→10, 10→11.

---

## Week 2, Day 3 — Added §9 "The Cost Argument — Quantization Is the Real Lever"
_Added: 2026-08-04. Driven by Yc's positioning goal: applying to startups / cost-conscious
employers, wanting to show strong outcomes at low cost with agent-assisted delivery._

### The framing that drives everything
At 1536 dims: **vector = 6,144 B, HNSW graph = ~136 B.** The graph is **2%** of the memory bill.
Compressing the *vector* is the only lever that matters. §7's ladder stopped at binary because
that's what pgvector offers; outside Postgres the ladder goes much further.

### Encodings and their real recall cost
| encoding | B/vector | vs fp32 | recall | +rerank |
|---|---|---|---|---|
| float32 + HNSW | 6,280 | 1× | 0.98 | — |
| float16 + HNSW | 3,208 | 2× | 0.98 | — |
| **int8 SQ + HNSW** | 1,672 | **4×** | **0.97** | — |
| PQ-96 (IVF) | 104 | 60× | 0.82 | 0.96 |
| **PQ-64 (IVF)** | 72 | **87×** | 0.75 | **0.94** |
| binary + rerank | 328 | 19× | 0.70 | 0.95 |

- **int8 scalar quantization is the underused winner** — 4× for ~1 point of recall, no rerank needed.
- **PQ mechanics** (whiteboard-ready): split 1536 dims into m=64 subvectors of 24 dims; k-means
  256 centroids per slice; store one centroid id (1 byte) per slice → 64 bytes. Distance becomes
  a table lookup per slice, so it's *faster* as well as smaller. OPQ adds a learned rotation first.
- **The omission to always call out:** PQ's good recall numbers assume a **rerank** against
  full-precision vectors. Raw PQ-64 is ~0.75 = unusable. So **compression applies to what must be
  in RAM, not to what must exist** — you still store the full vectors.

### The architectural recommendation (better than the common diagram)
Everyone draws "Postgres = metadata only, ANN engine = vectors." Better:
**keep the vector column, drop only the pgvector index.** An unindexed `vector` column costs
*disk, not RAM*, and Postgres keeps giving you: (1) durable source of truth for index rebuilds,
(2) transactional writes (Day 2's argument survives on the write path), (3) the rerank corpus via
b-tree point lookup. "Metadata only" throws all three away and still needs vectors stored durably
somewhere.

### Cost by scale (from `quantization_cost_calc.py`, all-in: infra + build + ops)
| scale | pgvector fp32 | best external | saving | verdict |
|---|---|---|---|---|
| 1M | $1,033/mo | $3,754/mo | **−264%** | pgvector wins; external costs 4× more |
| 10M | $3,577/mo | $3,764/mo | break-even | pgvector `halfvec` at $2,225 beats both |
| 100M | $29,016/mo | $3,880/mo | **87%** | external FAISS IVF-PQ wins decisively |
| 1B | $283,406/mo | $12,630/mo | **96%** | pgvector not deployable |

**The crossover sits between 10M and 100M** — below it the ~0.2 FTE ops load (~$3,000/mo) swamps
any infra saving. **Lead with the unflattering half in interviews:** "at our scale this would cost
more, so I wouldn't do it," then name where it inverts. An engineer who proposes the clever
architecture everywhere is a budget risk.

At 100M the PQ-64 bar is **$3,577 engineering / $303 infrastructure** — you're paying for people,
not RAM. That's the sentence that reframes the whole discussion.

### Library ranking
1. **FAISS + IVF-PQ** — memory is the constraint; best compression, GPU builds. Start here at 100M+.
2. **FAISS + HNSW** — high recall, index outside PG, corpus still fits RAM uncompressed.
3. **usearch** — modern C++, SIMD, mmap-able, lower footprint than hnswlib. Best small-team default.
4. **hnswlib** — plain HNSW, header-only.
5. **DiskANN** — 100M–1B+ where RAM binds; SSD-resident, hot nodes in memory.

Roll-your-own mmap (`vectors.bin` / `graph.bin` / `id_map.bin`): justified only for an access
pattern no library implements, or a C++ service where a dependency is unacceptable. **"I'd use
usearch and spend the time on the rebuild pipeline" is a better interview answer than describing
a custom index** — it shows you know where the hard part is.

### The agent-assisted claim — the credible version
`--build-weeks 1` vs 4 moves 100M cost from **$3,577 → $3,144/mo — only 12%**, because ongoing ops
dominates and doesn't shrink. **Agents compress build time, not operational load.** Claiming an
agent makes a service cheap to *own* won't survive an experienced interviewer; claiming it lets a
small team reach an architecture that used to need a bigger one is true and still impressive.
This is written into Day 3 §9 and as an interview drill in the ADR.

### New script
`week2/day3/quantization_cost_calc.py` — six encodings × four scale tiers, all-in cost, recall per
option, cheapest-meeting-SLO recommendation. Flags: `--dims`, `--recall-slo`, `--build-weeks`
(agent-assisted), `--reserved` (committed-use RAM).
Fixed during build: Postgres metadata sizing originally assumed the whole table stays hot; hydration
is *point lookups by id*, so only the PK index needs residency (~48 B/row, not row width).

### Also added
Day 6-7 ADR **§8 Cost-Optimised Variant** — the cost table, the keep-column-drop-index
recommendation, what the saving costs, and two interview drills ("how would you cut our vector
search bill?" and "you used AI coding tools — doesn't that make this less impressive?").
Day 3 now 12 sections; ADR now 9.

---

## Week 2, Day 3 — §10 The Sizing Ladder (added later)

**Request:** fold in the "practical rule of thumb" question order (documents → chunks → dimension →
change rate → QPS → latency SLA → choose architecture) and the interview answer that goes with it.

### The reframe applied
Kept the ordering and its punchline ("which vector database?" comes *last*), and extended the list
from 6 questions to 8. Two additions, both load-bearing:

- **Q5 — how selective are the filters?** The most-skipped question and the only one that can
  *delete the problem*. If every query is scoped (one card, one policy form, one category), the
  effective corpus is `N × selectivity` and exact search may beat any index.
- **Q8 — what does a wrong answer cost?** Sets the recall SLO, which picks the encoding, which is
  the biggest cost lever (per §9). Without it the ladder produces a size but no quality target.

Also reframed **Q2 (chunks/doc) and Q3 (dimension) as decisions, not inputs.** Both are usually
treated as handed down from the ingestion pipeline or the ML team. Halving chunk size doubles N;
MRL truncation halves the dimension. **Combined that is a 4–8× cost swing decided before a single
row is indexed** — larger than anything `ef_search` tuning recovers later.

### The finding that surprised me
Running the ladder for all three domains, **two of the three end with "no ANN index at all"** — and
one of them is the 2B-vector fintech case that looks like the hardest problem in the lesson:

| Domain | Filter → candidates | Verdict |
|---|---|---|
| Retail 4M | 2% → 80k | Exact **passes latency (20 ms) and fails throughput (41 cores @ 500 QPS)** → pgvector + HNSW on a replica |
| Insurance 200k | 5% → 10.2k | 2.6 ms, 0.21 cores → **no ANN index**; recall 1.0, nothing to rebuild, audit-reproducible |
| Fintech 2B | per-card → ~500 rows | 0.25 ms CPU, **1.02 cores @ 4,000 QPS** → partition by business key, exact inside |

**The retail asymmetry is the teaching point:** latency and throughput fail *separately*. Checking
only P99 is how you talk yourself into an architecture that collapses under load.

### New script
`week2/day3/sizing_ladder.py` — takes the eight answers, checks the filtered-exact path first
(latency and throughput separately), then picks encoding by recall SLO and pgvector-vs-external by
cost. Presets: `--domain retail|insurance|fintech-txn|fintech-docs`.

Its real value-add is the **sensitivity pass**: it perturbs each answer in both directions and
reports which question actually decided the outcome. **Across all four presets the load-bearing
answer is Q5, filter selectivity** — empirically confirming the thesis rather than asserting it.
When nothing flips (insurance, fintech-docs) it says so, which is itself worth saying in a design
review: stop arguing about inputs and start building.

Fixed during build:
- Encoding line contradicted the index line on the pgvector path (reported the cheapest possible
  encoding while recommending the halfvec+MRL ladder) → report per-path.
- Sensitivity flagged *wording* changes as architecture changes (two exact-search answers phrased
  differently). Added a coarse `kind` key and compare on that — false positives are what make a
  sensitivity table stop being believed.
- Q5 perturbed in one direction only; retail's interesting direction is *more* selective, fintech's
  is *less*. Now bidirectional, reporting whichever variant moves the answer furthest.
- Cost comparison alone recommended a 2.3 TB Postgres instance. Added `MAX_INSTANCE_RAM_GB = 768`
  feasibility guard **before** the cost comparison — past that ceiling the argument is availability,
  not cost.
- CPU-wall threshold for "needs a dedicated replica" was 2.0 cores, which let retail at 500 QPS slip
  through and contradicted §10's own domain table. Lowered to 1.0.

### The interview answer
Kept the supplied version verbatim in a "Good" box, then added a "Stronger" version. The two things
it adds are what a senior interviewer probes next: **a number attached to every claim** (7 GB per
million vectors at 1536 dims; ~5× from the ladder; crossover between 10M and 100M) and **the escape
hatch** — rule out partition-plus-exact *first*, because it is strictly the cheapest thing that works.

Day 3 is now **13 sections**, 12 quiz questions (added Q10 fintech-2B-is-an-illusion, Q11 the
latency/throughput asymmetry, Q12 the two questions that are really decisions).

---

# WEEK 3 — RAG ARCHITECTURE

Built in one session at Yc's request ("can we create the Week - 3"), Days 1 through 6-7.
Six lessons (0013–0018) plus six runnable stdlib-only scripts. Same method as Week 2:
build the artifact, run it, and let the verification correct the prose.

## Week 3, Day 1 — Chunking Strategies (`0013-chunking-strategies.html`)

### The refinement to Week 2's claim
Week 2 §10 Q2 said halving chunk size doubles infrastructure cost. True for the *index*, and
it is the smaller half of the bill. **Chunk size moves two costs in opposite directions:**

    smaller chunks -> more vectors        -> index cost UP
    smaller chunks -> fewer tokens/query  -> generation cost DOWN

Retail: at 1,600-token chunks generation is **81.6× infrastructure**. Total cost is
**U-shaped** with a minimum at 200 tokens ($15,488/mo); 1,600 tokens costs $47,732 — **3.1× the
optimum**, and it is exactly where an index-only cost model points you.

**Stated counterpoint:** the U-shape assumes fixed k. Under a fixed *context budget*
(`--fixed-context`) generation is flat and only the index moves, so large chunks win. Same
corpus, opposite conclusion — so the retrieval policy has to be stated before the cost number
means anything.

### The geometry result (derived, not asserted)
For answer span L, chunk C, overlap O, stride S = C−O:

    P(answer survives) = 1              if O >= L
                       = (C−L)/(C−O)    if O < L < C
                       = 0              if L >= C

**Overlap ≥ typical answer span guarantees no answer is ever split.** Not "overlap helps" — a
threshold. And below the span length, containment is *zero*: no reranker recovers a fact cut in
half at ingest.

### The recommendation that isn't the usual advice
**Buy containment with OVERLAP, not with chunk size.** Overlap is paid once at ingest as
storage; chunk size is paid again in every prompt. Insurance: **512 tok / 60% overlap = $984/mo**
beats 1,600 tok / 15% overlap at $1,927/mo — same guarantee, **half the cost**, signal density
58.6% vs 18.8%. Conventional "10–20% overlap" is right for short spans and badly wrong for long
ones; overlap should come from the measured p90 answer span.

### Per-domain answers (all different)
| | span | chunk / overlap | cost |
|---|---|---|---|
| Retail | 40 tok | 200 / 40 (20%) | $15,777/mo |
| Insurance | 300 tok | 512 / 307 (60%) | $984/mo |
| Fintech docs | 200 tok | 400 / 200 (50%) | $2,165/mo |

### `chunker_compare.py` findings
Four chunkers over a policy excerpt, scored on whether any single chunk answers three
cross-reference questions, swept across chunk budget:
- **structural answers all three from 30 tokens; recursive needs 70; fixed needs 90.**
- Structural wins **not because of better boundaries** but because it stamps the section path
  onto every chunk — the cheapest form of contextual retrieval, using context the document
  already provided.
- **Fixed is non-monotonic**: 3/3 at 90 tokens, 2/3 at 120. A noisy chunk-size sweep is a signal
  the chunker is fighting the document, not a signal to sweep harder. Also why "we tuned chunk
  size on our eval set" can mean "we overfit to which sentences landed together."

### Also covered
Overlap costs retrieval diversity too (near-duplicates crowd top-k) → span dedup + adjacent
merge must ship *with* high overlap. `content_hash` is the metadata field people omit and regret
— without it every chunking change is a full re-embed. Small-to-big resolves the §1 tension:
index 200-token children, return 1,600-token parents via a **b-tree point read**, not a second
vector search.

12 sections, 8 quiz questions, interactive chunk-economics explorer, 3 animated SVGs.

## Week 3, Day 2 — Dense vs Sparse vs Hybrid (`0014-dense-sparse-hybrid.html`)

Differentiated from Week 2 Day 4 (which was *production* hybrid: routing, RRF weights, eval
traps). This is the layer underneath — what each arm computes and therefore what it cannot
retrieve.

### The measured result, which is not the one I expected
`bm25_vs_tsrank.py` implements BM25 and the identical scorer with IDF deleted. First attempt
showed **no divergence at all** — the toy corpus had no decoys, and my "common" terms appeared in
only 4 documents so they weren't common. After fixing the generator:

**BM25 holds rank 1 in every cell. The no-IDF scorer loses the answer from 6 query terms onward,
at EVERY corpus size.** So the trigger is **query length, not corpus size**. Corpus size sets the
*magnitude* — IDF spread runs 4.8× at N=5 to 66× at 4M.

Why it survives to production: dev corpora are curated and lack decoys; manual testing uses short
queries; and hybrid fusion lets the dense arm cover for it. The class at risk is
`NATURAL_LANGUAGE` — exactly the one teams assume dense is carrying anyway.

### Fixes, in order
1. `setweight()` field weighting (A/B/D) — doesn't restore IDF but approximates it where
   discriminators live in specific columns. Cheapest by far.
2. `ts_stat()` → materialised term→idf table. Real BM25 in SQL, and a refresh job you now own.
3. Let dense carry it — legitimate only if you've *measured* that it does.

**If fused ≈ dense-only, the lexical arm is contributing nothing** — a hybrid system paying for
two arms and running one. Measure the arms separately, per class.

### Also
BM25 derived (k1 saturation, b length-norm, IDF). **Tuning note specific to Day 1: fixed-size
chunks removed the length variance `b` exists to correct, so b→0.3 or 0 is often right on a
chunked corpus and is almost never tested.** RRF's one real weakness: it discards score
magnitude, so it cannot express "nothing good here" → pre-fusion score floor. SPLADE covered and
rejected *for Postgres* — adopting it means adopting a search engine, which breaks the operational
argument that kept you in Postgres.

## Week 3, Day 3 — Reranking (`0015-reranking.html`)

### The framing
**A reranker improves precision, never recall.** End-to-end quality is bounded by `recall@N`.
If recall@100 = 0.72, 28% of queries are unanswerable regardless of reranker quality.

### The sizing result
**Cost is linear in N; the recall curve saturates. That guarantees an efficiency optimum strictly
below the latency limit** — so "set N to whatever fits the budget" is reliably wrong.

Retail: **N=50 at $523 per +0.01 of ceiling** vs N=200 (the largest that fits) at **$1,546** —
3× worse per unit of quality. The `$ per +0.01` column is what makes the decision, not recall@N.

`rerank_economics.py` originally recommended the largest affordable N; corrected to pick on
marginal efficiency.

### The LLM-reranker reality check (corrected my own domain note)
I had written that insurance's 2-second budget means "even an LLM reranker fits." The script
disproved it: **at 512-token chunks even N=25 needs 3.2 s**. N=10 fits but costs $16,589/mo vs
$584 for a cross-encoder at N=100 — 28× the price for a *worse* ceiling (0.62 vs 0.89).
**The binding constraint is prompt length, not QPS.** Note rewritten.

### Also
GPU granularity: insurance costs the same $584 at N=10 and N=100 because both fit one GPU —
below a machine's capacity the efficiency argument *inverts*. Routing (rerank only
`NATURAL_LANGUAGE` + `AMBIGUOUS`) drops retail from 13 GPUs to 8, **$7,592 → $4,672/mo, 38%, zero
quality loss**. Also added: a feasibility guard so cost comparison can't recommend an
undeployable configuration.

## Week 3, Day 4 — RAG Evaluation (`0016-rag-evaluation.html`)

The angle that's usually missing is statistical, and it's the strongest artifact of the week.

### What the harness demonstrates
Simulated data **with known ground truth**. System B is genuinely better by **+0.0277 nDCG**.
The 50-query eval measured **−0.0208 — the wrong sign** — and correctly reported NOT significant
(MDE at n=50 is 0.0628, 2.3× the true effect).

- "Compare the means, ship the winner" → **ships A**, rejecting a real improvement and recording
  a wrong-signed number in the decision doc.
- Paired bootstrap → "not significant", which is the only honest answer available from 50 queries.
- **The bootstrap didn't find the truth. Nothing could. It stopped you asserting a falsehood.**

At n=2000 it converges correctly: +0.0211, CI excludes zero.

### The variance insight
Per-query paired nDCG SD is **~0.16, not 0.02–0.05**, because a retrieval change leaves most
queries untouched and **flips a minority hard** — a query either surfaces the right chunk or it
doesn't. Since n scales as SD², assuming small noise understates the required eval set by an
order of magnitude.

To detect +0.03: **~220 queries**, or **~537** to resolve it inside `NATURAL_LANGUAGE` (41% of
traffic). Most teams have 40–50.

Two simulation bugs fixed during the build: noise model was unrealistically smooth (made the eval
look far more powerful than it is), then a hard clamp introduced systematic negative bias large
enough to flip the population effect's sign — moved to a **logit-space shift**, which is bounded
by construction and still produces a real ceiling effect. Script now measures and reports the
true population effect over 200k queries rather than assuming the input parameter is the truth.

### Also
Refusal correctness needs its own negative eval set — no other part of the suite contains
unanswerable questions. **LLM-judge bias is systematic, not random, so it does NOT average out**:
10,000 judged queries can be more confidently wrong than 50 human-labelled ones.

## Week 3, Day 5 — Failure Modes (`0017-failure-modes.html`)

Focus: **conflicting sources** — the failure that passes every Day 4 metric.

### The mechanism
Revised documents produce near-identical versioned chunks. Near-identical text is near-identical
in embedding space, so a retrieval that finds one finds them all. The model gets three versions,
is told nothing about which is in force, and picks one.

Groundedness scores it **GROUNDED** — because the claim genuinely *is* supported by a real,
correctly-quoted retrieved chunk.

### The result worth remembering
**Reranking causes this.** Insurance 36% → 100%; **retail 0.1% → 97.5%**. A cross-encoder scores
each candidate against the query independently, and every version answers about equally well, so
it promotes all of them. Day 3's precision win and this failure are the same mechanism from two
sides. **The effective-date filter must ship BEFORE the reranker** — otherwise you manufacture a
failure mode that did not previously exist.

### Detection vs prevention
"Instruct the model to prefer the newest" is a **detection** control — it can only choose among
what was retrieved, and in **28.9%** of insurance queries the context held a superseded version
*without* the current one. The date filter is a **prevention** control. Audit committees ask
which one you have.

**As-of date is an input, not `today()`.** A 2023 claim is adjudicated under the 2023 wording; a
dispute under the rules in force at transaction time. Hardcoding today fails quietly, in the
direction that loses litigation.

Two bugs fixed: version-aware dedup was keeping the *highest-scoring* version rather than the
newest (retrieval score says nothing about what's in force), and the candidate pool was
degenerate so versions always swept top-k.

Also: relevance graders shown three versions mark all three relevant — **the labelling process
itself erases the distinction the eval needs**. Catching this requires purpose-built questions
whose answers changed between versions, graded against a stated as-of date.

## Week 3, Day 6-7 — Checkpoint / ADR-003 (`0018-checkpoint-retrieval-redesign.html`)

Full pipeline redesign with five decisions, each traced to its number, plus "what would change
our mind" for each. Rollout order with two non-obvious dependencies (date filter before
reranker; span dedup *with* the overlap increase, not after).

### The cost line that reframes it
Retail: generation **$10,854 (53%)**, index **$4,910 (24%)**, reranking $4,672, embed $13 —
**$20,449/mo total, down from $27,574** despite *adding* a reranker, because smaller chunks
shrink the prompt faster than they grow the index.

**Week 2 spent five days on the index because that's where the interesting engineering is. Week 3
found the prompt is where the money is.** Both matter: the index is where you break the system,
the prompt is where you bankrupt it.

### The honest closing
Four of the five ADR decisions rest on **one measurement not yet made properly** — the p90
answer-span distribution (50 examples). Stated explicitly as the thing to revisit first, with the
reasoning that "here's the model, here's what it recommends, and here's the input I'd want more
evidence for" beats a confident recommendation with an unexamined premise.

Four interview drills as `<details>` blocks.

## Verification
9 SVGs parse, all 6 scripts run, all flags exercised, 58 sections and 37 quiz questions across
the six lessons, only the intentional Week 4 forward link
(`week4/day1/0019-langgraph-fundamentals.html`) unresolved.

## Handover prompt for Week 4, Day 1

> I'm working through a 12-week AI Data Architect prep curriculum in `Claude/`.
> 1. Please review `Claude/mission.md`, `Claude/dataPrep.md`, and `Claude/NOTES.md`
>    (Weeks 2 and 3) for my 14-year Data Engineering background and three-domain target
>    (retail / life insurance / fintech).
> 2. Completed so far:
>    - **Week 1 (Embeddings):** vector physics → distance metrics → OpenAI vs open-source →
>      fine-tuning → justification framework → ADR-001.
>    - **Week 2 (Vector Databases):** pgvector internals → managed vector DBs and the dual-write
>      problem → five scale walls, mitigation ladder, quantization economics, the eight-question
>      sizing ladder → production hybrid search → operational simplicity → ADR-002.
>    - **Week 3 (RAG Architecture):** chunking as a joint quality/cost decision → dense vs sparse
>      vs hybrid and the missing IDF → reranking economics → evaluation statistics → failure
>      modes and version conflicts → ADR-003.
> 3. Today's Topic (Week 4, Day 1): LangGraph fundamentals — state machines vs simple chains,
>    why cycles and conditional routing matter. Same treatment: concrete numbers, three-domain
>    worked examples, runnable stdlib-only companion script, animated visuals for anything
>    involving memory, cost, or comparison.
>    - Reference lesson artifact: `Claude/week4/day1/0019-langgraph-fundamentals.html`
> 4. Carry forward:
>    - Week 3 Day 4 established that **most eval sets cannot resolve the effects being argued
>      about** — apply the same power discipline to any agent-behaviour evaluation.
>    - Week 3 Day 5 established **detection vs prevention controls** — useful lens for
>      human-in-the-loop design (Week 4 Day 3).
>    - Agent state persistence in Postgres is Week 4 Day 5 and should connect back to ADR-002.
>
> Let's dive into Week 4, Day 1!
