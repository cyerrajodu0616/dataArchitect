# Add GitHub Pages lesson index

## Feature context

GitHub Pages currently publishes the repository, but the repository root has no `index.html`. Users therefore need to remember a full lesson URL such as `Claude/week1/day1/0001-what-is-an-embedding.html`. Add a mobile-friendly root landing page that links to every existing HTML lesson.

## Files

- Create `index.html` at repository root. This is a new file, so there is no before block or existing line number.
- Do not modify any lesson page or shared stylesheet.

## Exact implementation

Create `index.html` beginning at line 1 with exactly:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Data Architect Learning Path</title>
  <style>
    :root { color-scheme: light; --ink:#17202a; --muted:#667085; --paper:#f6f7f9; --card:#fff; --accent:#3157a4; --line:#e4e7ec; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--paper); line-height:1.5; }
    header { padding:3.5rem 1.25rem 2.5rem; color:#fff; background:linear-gradient(135deg,#18345f,#3157a4); }
    header div, main { width:min(100%,70rem); margin:auto; }
    h1 { margin:0 0 .5rem; font-size:clamp(2rem,6vw,3.5rem); line-height:1.1; }
    header p { max-width:42rem; margin:0; color:#e7ecf7; font-size:1.05rem; }
    main { padding:2rem 1.25rem 4rem; }
    section + section { margin-top:2.5rem; }
    h2 { margin:0 0 1rem; font-size:1.45rem; }
    .lessons { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,17rem),1fr)); gap:.85rem; }
    a { display:block; min-height:100%; padding:1rem; color:inherit; text-decoration:none; background:var(--card); border:1px solid var(--line); border-radius:.75rem; box-shadow:0 1px 2px rgb(16 24 40 / 5%); }
    a:hover, a:focus-visible { border-color:var(--accent); box-shadow:0 4px 12px rgb(49 87 164 / 14%); transform:translateY(-1px); }
    a:focus-visible { outline:3px solid rgb(49 87 164 / 25%); outline-offset:2px; }
    .number { display:block; margin-bottom:.3rem; color:var(--accent); font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
    .title { font-weight:650; }
    @media (prefers-reduced-motion:no-preference) { a { transition:border-color .15s,box-shadow .15s,transform .15s; } }
  </style>
</head>
<body>
  <header><div><h1>Data Architect Learning Path</h1><p>Embeddings, vector databases, retrieval-augmented generation, and agent architecture—from fundamentals to production decisions.</p></div></header>
  <main>
    <section><h2>Week 1 · Embeddings</h2><div class="lessons">
      <a href="Claude/week1/day1/0001-what-is-an-embedding.html"><span class="number">Lesson 0001</span><span class="title">What an Embedding Actually Is</span></a>
      <a href="Claude/week1/day2/0002-distance-metrics.html"><span class="number">Lesson 0002</span><span class="title">Distance Metrics</span></a>
      <a href="Claude/week1/day3/0003-openai-vs-opensource-embeddings.html"><span class="number">Lesson 0003</span><span class="title">OpenAI vs Open-Source Embeddings</span></a>
      <a href="Claude/week1/day4/0004-fine-tuning-embeddings.html"><span class="number">Lesson 0004</span><span class="title">Fine-Tuning vs Off-The-Shelf</span></a>
      <a href="Claude/week1/day5/0005-embedding-model-justification.html"><span class="number">Lesson 0005</span><span class="title">Embedding Model Justification</span></a>
      <a href="Claude/week1/day6-7/0006-checkpoint-fine-tuning-1pager.html"><span class="number">Lesson 0006</span><span class="title">Week 1 Architecture Checkpoint</span></a>
    </div></section>
    <section><h2>Week 2 · Vector Databases</h2><div class="lessons">
      <a href="Claude/week2/day1/0007-pgvector-fundamentals.html"><span class="number">Lesson 0007</span><span class="title">pgvector Fundamentals</span></a>
      <a href="Claude/week2/day2/0008-pinecone-weaviate-milvus.html"><span class="number">Lesson 0008</span><span class="title">Pinecone, Weaviate, and Milvus</span></a>
      <a href="Claude/week2/day3/0009-pgvector-scale-limits.html"><span class="number">Lesson 0009</span><span class="title">Where pgvector Breaks</span></a>
      <a href="Claude/week2/day4/0010-hybrid-search-production.html"><span class="number">Lesson 0010</span><span class="title">Hybrid Search in Production</span></a>
      <a href="Claude/week2/day5/0011-operational-simplicity.html"><span class="number">Lesson 0011</span><span class="title">Operational Simplicity</span></a>
      <a href="Claude/week2/day6-7/0012-checkpoint-pgvector-justification.html"><span class="number">Lesson 0012</span><span class="title">pgvector Scale Checkpoint</span></a>
    </div></section>
    <section><h2>Week 3 · Retrieval-Augmented Generation</h2><div class="lessons">
      <a href="Claude/week3/day0/0013a-rag-from-zero.html"><span class="number">Lesson 0013A</span><span class="title">RAG From Zero</span></a>
      <a href="Claude/week3/day1/0013-chunking-strategies.html"><span class="number">Lesson 0013</span><span class="title">Chunking Strategies</span></a>
      <a href="Claude/week3/day2/0014-dense-sparse-hybrid.html"><span class="number">Lesson 0014</span><span class="title">Dense vs Sparse vs Hybrid</span></a>
      <a href="Claude/week3/day3/0015-reranking.html"><span class="number">Lesson 0015</span><span class="title">Reranking</span></a>
      <a href="Claude/week3/day4/0016-rag-evaluation.html"><span class="number">Lesson 0016</span><span class="title">RAG Evaluation</span></a>
      <a href="Claude/week3/day5/0017-failure-modes.html"><span class="number">Lesson 0017</span><span class="title">RAG Failure Modes</span></a>
      <a href="Claude/week3/day6-7/0018-checkpoint-retrieval-redesign.html"><span class="number">Lesson 0018</span><span class="title">Retrieval Pipeline Checkpoint</span></a>
    </div></section>
    <section><h2>Week 4 · Agent Architecture</h2><div class="lessons">
      <a href="Claude/week4/day1/0019-langgraph-fundamentals.html"><span class="number">Lesson 0019</span><span class="title">LangGraph Fundamentals</span></a>
      <a href="Claude/week4/day2/0020-framework-comparison.html"><span class="number">Lesson 0020</span><span class="title">Framework Comparison</span></a>
      <a href="Claude/week4/day3/0021-human-in-the-loop.html"><span class="number">Lesson 0021</span><span class="title">Human in the Loop</span></a>
      <a href="Claude/week4/day4/0022-error-handling-retries.html"><span class="number">Lesson 0022</span><span class="title">Error Handling and Retries</span></a>
      <a href="Claude/week4/day5/0023-state-persistence.html"><span class="number">Lesson 0023</span><span class="title">State Persistence</span></a>
      <a href="Claude/week4/day6-7/0024-checkpoint-agent-state-graph.html"><span class="number">Lesson 0024</span><span class="title">Agent State Graph Checkpoint</span></a>
    </div></section>
  </main>
</body>
</html>
```

## Verification

1. Confirm all 25 `href` targets exist in the repository.
2. Serve the repository root locally and confirm `/` displays the index.
3. Check the page at phone width (390 px), iPad width (768 px), and desktop width (1440 px).
4. Open at least one link from each week and confirm it resolves without a 404.
5. Confirm keyboard focus is visible and `prefers-reduced-motion` is respected.

## Constraints and gotchas

- Keep links relative so the page works both locally and at the GitHub Pages project path `/dataArchitect/`.
- Do not use a leading slash in lesson URLs; that would point at the account-level site root.
- Do not introduce a build system or external dependency for this static page.
- Publishing still requires committing and pushing the new file to the `main` branch.
