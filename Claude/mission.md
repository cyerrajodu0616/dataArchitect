# Mission
 
## Who
Yc — data engineer, ~14 years experience (SQL, data modeling, PySpark, Hive).
 
## Why
Transitioning career over the next 3–6 months from Data Engineer → **AI/Data Platform Architect**
(secondary: AI/ML Engineer), targeting the **retail**, **life insurance / insurtech**, and
**fintech** domains. This is the singular focus for the next 3 months — deviations should be
redirected back to this plan.

_Domain scope widened 2026-08-03 at Yc's request. Retail remains the primary worked example;
life insurance and fintech are carried alongside it in every lesson. Insurance/insurtech is the
closest match to Yc's current working context, and it exercises constraints retail does not —
PHI/PII handling, state-by-state form variation, and decision auditability. Fintech contributes
the opposite scale profile (billions of transaction embeddings, real-time freshness), which is
useful precisely because it is the domain where the pgvector answer flips._
 
## The differentiator
Deep production data engineering experience is the wedge against typical GenAI engineers who lack
it. Every lesson should reinforce this: not "learn AI from scratch," but "translate 14 years of
data engineering judgment into AI/data platform architecture judgment."
 
## The vehicle
A personal agent project (LangGraph + Postgres/pgvector + OpenAI embeddings), built primarily with
Claude Code, used as the running example for every concept. Minimal toolset by design — net-new
additions are `rank_bm25` and `ragas` only; everything else studied conceptually, not deployed.
 
## The structure
A 12-week plan (see project file `AI-Data-Architect-Prep-Plan.md`), four phases:
- Weeks 1–4: Embeddings & foundational "why" knowledge
- Weeks 5–8: Scaled data pipelines for AI
- Weeks 8–9: System design practice
- Weeks 10–12: Portfolio and interview positioning
Each week ends in a tangible deliverable (justification doc, redesigned pipeline, architecture
diagram). Sessions are referenced as "Week X, Day Y."
 
## What "done" looks like
By end of Week 12: a portfolio-ready architecture writeup of the personal agent, a repositioned
resume/LinkedIn, and interview-ready command of the "why this over that" decisions log —
culminating in an active job search for AI/Data Platform Architect roles in retail.
 
## Success criteria for this teach workspace specifically
Lessons build **storage strength**, not just fluency — Yc should be able to explain these concepts
cold, unprompted, in an interview room, weeks after the lesson. Every lesson ties back to a real
decision in the personal agent project or a retail scenario, not abstract theory.