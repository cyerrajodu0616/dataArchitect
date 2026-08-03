# Notes

- Prefers going deep with concrete, worked examples over abstract explanation.
- Strong existing intuition from data engineering (warehouse tables, feature vectors, distance
  concepts) — use these as anchors/analogies rather than starting from zero.
- Minimal toolset preference — don't introduce new tools/libraries unless there's a clear
  deployment reason. Study broadly, deploy narrowly.
- Retail domain flavor should show up in examples wherever possible (product catalogs, SKUs,
  customer search, personalization).
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
