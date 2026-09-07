# Deepen Week 4 Days 2–5 and the final checkpoint

## Context

Day 1 now moves from conceptual explanation to a real, deterministic LangGraph program with tests. Days 2–5 still stop mainly at pseudocode and quantitative calculators, and the final checkpoint is an ADR rather than a runnable integrated system. Bring the remaining Week 4 lessons to the same depth without expanding Day 0, which remains the orientation lesson.

Current LangGraph APIs were verified on 2026-09-06 against the official Graph API, interrupts, persistence, durable execution, and fault-tolerance documentation. The repository already declares `langgraph>=1.2.11,<2.0.0` in `pyproject.toml`; do not add framework or database dependencies.

## Files and exact changes

### Day 2 — framework choice

#### `Claude/week4/day2/0020-framework-comparison.html`

1. Correct the current graph pseudocode at lines 86–91. The current final line is:

```html
result = g.compile(checkpointer=pg).invoke(state, thread_id=case_id)</div>
```

Replace it with the supported config form:

```html
result = g.compile(checkpointer=pg).invoke(
    state,
    config={"configurable": {"thread_id": case_id}},
)</div>
```

2. Insert `7. Compare the Architectures by Running Them` immediately before current line 251 `<h2>7. Companion Script</h2>`. Cover one deterministic refund request executed as:
   - raw explicit loop;
   - fixed forward pipeline;
   - real LangGraph state machine;
   - a deliberately scoped crew-style coordinator simulation, clearly labeled as a control-flow model rather than CrewAI/AutoGen code.
3. Compare observable path, ownership of routing, loop exit, pause/resume readiness, trace shape, and change surface. Explicitly state that this is a structural lab, not a model-quality or performance benchmark.
4. Add a change exercise: introduce human approval and show which implementation boundaries move.
5. Renumber current sections 7 and 8 to 8 and 9; update the companion table and command block for the new lab and tests.

Insertion anchor copied from lines 243–253:

```html
  <div class="fin">
    <div class="dhead">Fintech — disputes</div>
    <p><span class="kv">loops: 1 &#183; human pause: threshold<br>durable: days &#183; audit: yes</span></p>
    <p><strong>Graph.</strong> Regulatory deadlines mean a run must survive restarts, and high-value disputes pause for a human.</p>
    <p>The deadline is a hard external clock, which makes durable state a compliance property, not a convenience.</p>
  </div>
</div>

<h2>7. Companion Script</h2>

<table class="matrix-table">
```

#### Create `Claude/week4/day2/framework_shapes_lab.py`

Use the Day 1 deterministic intent/order data. Implement `run_raw`, `run_chain`, `run_graph`, and `run_crew_simulation` returning a common `RunResult` dataclass with `decision`, `path`, `state`, and `routing_owner`. The graph path must use the real `StateGraph`; the other three must be plain Python. Add CLI `--approach`, `--request`, `--order-id`, `--require-approval`, and `--trace`. Do not import CrewAI or claim the simulation measures CrewAI.

#### Create `Claude/week4/day2/test_framework_shapes_lab.py`

Use `unittest` to verify common clear-request outcomes, the fixed chain's explicit inability to clarify, raw/graph clarification paths, routing ownership labels, graph state trace, and the change surface when approval is required. Ensure root-directory and Day 2-directory test invocation both work.

### Day 3 — human approval lifecycle

#### `Claude/week4/day3/0021-human-in-the-loop.html`

1. Replace the illustrative static `interrupt_before` example at current lines 233–245 with the current dynamic interrupt pattern: an approval node calls `interrupt(JSON_serializable_payload)` and the caller resumes the same `thread_id` with `Command(resume=decision)`.
2. Replace current lines 247–250, which say:

```html
<div class="anchor">
<div class="label">The thing to notice</div>
<code>thread_id</code> is doing all the work — it's the primary key of a durable row holding the whole state. Resuming is a <code>SELECT</code>, not magic.
<p style="margin-top:0.7rem;margin-bottom:0;">Which is exactly why Day 2's damage table scored "pause for a human" and "survive a crash" identically: <strong>they're the same requirement.</strong> Both need durable state resumable at an exact node. Day 5 sizes what that costs in Postgres.</p>
</div>
```

with a precise explanation: `thread_id` is the stable application-level cursor used by the checkpointer; checkpoint namespaces/IDs and version metadata identify stored snapshots. It is not, by itself, a complete schema or concurrency strategy.
3. Reword all statements at current lines 108, 161, and 177 that present overflow as automatic approval. Treat auto-approval as one dangerous fail-open policy being modeled, not the default. Add fail-closed, explicit expiry/escalation, and domain-specific timeout policy alternatives.
4. Insert `9. Build the Approval Lifecycle` before current line 300. Cover JSON-safe approval payloads, actor identity, role authorization, approve/reject/edit decisions, expiry, stale-state revalidation, optimistic version checks, audit events, duplicate/concurrent responses, and why code before `interrupt()` must be idempotent.
5. Add a state transition table: `PENDING → APPROVED|REJECTED|EXPIRED|STALE`, with no transition out of terminal states.
6. Renumber current sections 9 and 10 to 10 and 11; update companion table, commands, quiz cross-references, and final ask-agent copy.

Insertion anchor copied from lines 276–302:

```html
<h2>8. What This Model Does Not Price</h2>

<table class="matrix-table">
```

Insert the new section after the complete Section 8 table and immediately before:

```html
<h2>9. Companion Script</h2>
```

#### Create `Claude/week4/day3/approval_interrupt_lab.py`

Build a deterministic LangGraph approval workflow using `InMemorySaver`, `interrupt()`, `Command(resume=...)`, and stable `thread_id`. Define a JSON-safe `ApprovalRequest` payload and validated response actions `approve`, `reject`, and `edit`. Record append-only audit events containing actor, action, version, and timestamp supplied by a deterministic clock. Implement fail-closed expiry, stale-version rejection, and first-terminal-response-wins behavior. No network, model, or wall-clock dependency.

#### Create `Claude/week4/day3/test_approval_interrupt_lab.py`

Test initial interrupt payload, authorized approval, rejection, edit-then-approve behavior, unauthorized actor, expired approval, stale version, duplicate response, concurrent-response winner, audit completeness, and same-thread resume. Use standard-library `unittest`.

### Day 4 — production reliability mechanics

#### `Claude/week4/day4/0022-error-handling-retries.html`

1. At current lines 47–57, qualify the multiplication model: `p^N` assumes independent, equally reliable required steps. Add correlated dependency, shared-model, and shared-configuration failures, and explain that production reliability must be measured by path/step class rather than inferred from one average.
2. At current lines 146–215, replace “with a key, both are safe” and zero-duplicate absolutes with the conditions required: stable logical-operation key, receiver-side unique enforcement, atomic result storage, sufficient key retention, and identical request semantics for reuse.
3. Insert `8. Build the Reliability Boundary` immediately before current line 278. Cover typed failure classification, retry budgets, attempt deadlines, full jitter with deterministic injection for tests, idempotency record states, circuit breaker `CLOSED/OPEN/HALF_OPEN`, dead-letter envelope, replay versus restart, and structured telemetry.
4. Explain that model-output repair is a new attempt with validation feedback, not a transport retry.
5. Renumber current sections 8 and 9 to 9 and 10; update companion table, commands, and final copy.

Insertion anchor copied from lines 266–280:

```html
<h2>7. What to Build, In Order</h2>

<table class="matrix-table">
```

Insert after that complete table and immediately before:

```html
<h2>8. Companion Script</h2>
```

#### Create `Claude/week4/day4/resilient_refund_lab.py`

Implement a standard-library deterministic reliability boundary with `FailureKind`, a typed `AttemptResult`, a receiver-side idempotency store, retry budget, injected sleeper/random source, full-jitter delay calculation, three-state circuit breaker, dead-letter record, and replay-by-state. Include simulated transient, permanent, ambiguous-after-commit, malformed-model-output, and correlated-failure modes. Do not perform real sleeps or network calls.

#### Create `Claude/week4/day4/test_resilient_refund_lab.py`

Test retryable versus permanent classification, stable key reuse, ambiguous-after-commit deduplication, conflicting payload rejection for one key, jitter bounds, breaker transitions, dead-letter contents, replay behavior, and correlated-failure trip behavior. Add a test documenting the independence assumption in the existing probability function.

### Day 5 — durable state and concurrency

#### `Claude/week4/day5/0023-state-persistence.html`

1. Replace the heading and claim at current lines 54–75. Preserve the illustrative table as a teaching simplification but state that a production checkpointer also needs namespace/checkpoint identity, parent lineage, version/channel metadata, pending writes, and serialization metadata. `thread_id` is a lookup cursor, not “the primary key” by itself.
2. Correct current lines 197–210. Colocation in Postgres makes a shared transaction possible only when business mutation and checkpoint use the same connection and explicit transaction boundary. It does not atomically cover an external payment/insurance API; those need idempotency/outbox/reconciliation patterns.
3. Qualify “delta checkpoint” guidance: distinguish the lesson's sizing model from the installed checkpointer's actual storage representation; do not claim LangGraph uses this custom delta/anchor scheme unless demonstrated.
4. Insert `6. Build and Operate a Checkpointer` before current line 247. Cover `InMemorySaver` for tests only, production `PostgresSaver.from_conn_string(...)` plus one-time `setup()`, stable thread IDs, checkpoint history/state inspection, optimistic concurrency, duplicate resumes, schema/version migration, serializer safety, encryption and PII trimming, retention/partitioning, restore drills, and observability.
5. Add an explicit concurrency timeline where two workers resume the same approval and only one wins.
6. Renumber current sections 6 and 7 to 7 and 8; update companion table, commands, quiz references, and final copy.

Before block copied from current lines 72–75:

```html
<div class="anchor">
<div class="label">That's the whole checkpointer</div>
A write after each node, and a <code>SELECT ... ORDER BY step DESC LIMIT 1</code> to resume. Day 3's <code>thread_id</code> is just the primary key. Once you've seen this, "the framework handles durable resumption for you" stops sounding like magic and starts sounding like <em>a table you didn't have to write</em> — which is the correct amount of credit to give it.
</div>
```

After block must retain the beginner analogy but label the table as a minimum mental model and enumerate the missing production metadata listed above.

#### Create `Claude/week4/day5/checkpoint_lifecycle_lab.py`

Use LangGraph `InMemorySaver` to demonstrate checkpoint creation, `thread_id`, final state, and history inspection without external infrastructure. Add a standard-library `CheckpointLeaseStore` simulation with optimistic version, lease owner/expiry, compare-and-swap resume, terminal-state protection, schema version, and deterministic migration from schema v1 to v2. Use injected time and no real sleep/network/database.

#### Create `Claude/week4/day5/test_checkpoint_lifecycle_lab.py`

Test checkpoint history, thread isolation, same-thread continuation, one-winner concurrent lease, expired lease takeover, stale-version rejection, terminal protection, v1→v2 migration, and state trimming of raw tool payloads/PII.

### Days 6–7 — integrated checkpoint

#### `Claude/week4/day6-7/0024-checkpoint-agent-state-graph.html`

1. Correct current line 124 from “Duplicate side effects go to zero” to the conditional guarantee defined in Day 4.
2. Correct current line 134 so approval-queue timeout is an explicit fail-open scenario, with the chosen final design defaulting fail-closed and escalating/expiring.
3. Correct transaction claims at current lines 208 and 237 using the Day 5 shared-transaction boundary language.
4. Qualify node reliability math at current lines 114, 132, 151, 205–206 with its independence/equal-rate assumption.
5. Add `6. Execute the Architecture End to End` before current line 200. Walk through four complete traces: ambiguous-to-tracking, approved refund, rejected/expired high-value refund, and ambiguous-after-commit retry safely deduplicated. For every trace show state delta, route, checkpoint/interrupt boundary, side effect, audit event, and terminal result.
6. Add a production-readiness matrix: control, implementation, test, metric, alert, owner.
7. Renumber current Interview Drills and Companion Scripts to sections 7 and 8. Update companion table with all new labs and the integrated checkpoint program.

Insertion anchor copied from current lines 194–202:

```html
<div class="caution">
<div class="label">The honest summary of this week</div>
Three of the six ADR decisions depend on <strong>one number nobody has measured: the agent's actual error rate.</strong> The approval threshold, the retry economics, and the value of the review queue all move with it.
<p style="margin-top:0.7rem;margin-bottom:0;">The models are sound and their sensitivity is explicit, which is the useful state to be in — but "we assumed 4%" is the sentence to volunteer rather than have extracted. Sampling 200 actions and grading them is a week of work and it firms up half this document.</p>
</div>

<h2>6. Interview Drills</h2>
```

#### Create `Claude/week4/day6-7/production_agent_checkpoint.py`

Compose the Week 4 concepts into one deterministic LangGraph program using typed state/reducers, explicit routing, `InMemorySaver`, dynamic `interrupt`, validated resume commands, stable idempotency keys, the Day 4 reliability boundary, audit events, fail-closed expiry, and terminal results. Reuse code through imports from the adjacent lesson lab files only where imports remain clear and deterministic; otherwise use small local adapters rather than copy large implementations. CLI scenarios: `tracking`, `refund-approved`, `refund-rejected`, `refund-expired`, and `ambiguous-commit`.

#### Create `Claude/week4/day6-7/test_production_agent_checkpoint.py`

Test all five CLI-equivalent scenarios, trace order, checkpoint/interrupt boundaries, single side effect under ambiguous retry, terminal-state protection, audit trail, and no side effect before approval.

## Verification for the whole change

1. Run every existing Week 4 companion script to prove no regression.
2. Run every new `unittest` file from repository root and its lesson directory.
3. Run every documented CLI scenario; no command may require an API key, database, network, or real sleep.
4. Parse all Week 4 HTML files; require balanced tags, sequential headings, working navigation links, and intact quiz scripts.
5. Check every companion path mentioned in HTML exists.
6. Run `uv lock --check`, `git diff --check`, and compile all new Python files.
7. Verify no unrelated files under `Claude/week5/`, `scripts/build_interview_profile.py`, or the two interview-profile prompt files were modified.

## Constraints and gotchas

- Day 0 remains unchanged.
- Preserve the existing calculators and their CLI behavior.
- Use deterministic clocks, random sources, and failure schedules in tests.
- `InMemorySaver` is educational/test-only; never present it as restart-safe production persistence.
- Interrupt payloads must be JSON-serializable, and code before `interrupt()` must be idempotent because a node reruns on resume.
- `Command(resume=...)` resumes with the same stable `thread_id`.
- Default approval timeout behavior is fail-closed; alternative fail-open economics may be discussed but not normalized as safe.
- Idempotency guarantees depend on stable keys and receiver-side atomic enforcement.
- No CI/CD or deployment files may be touched.

## Non-material consistency cleanup during implementation

The final review found that the preserved calculator/chooser output repeated four claims corrected in the HTML. Update wording only, without changing calculator behavior or CLI arguments, in:

- `Claude/week4/day2/framework_chooser.py`: current `configurable.thread_id` invocation and dynamic `interrupt()` terminology.
- `Claude/week4/day3/approval_economics.py`: label overflow auto-approval as the deliberately unsafe fail-open policy being priced; production defaults fail-closed.
- `Claude/week4/day4/reliability_math.py`: state the receiver-side atomic enforcement conditions behind the zero-duplicate model.
- `Claude/week4/day5/checkpoint_sizing.py`: qualify trimming, transactionality, delta encoding, and the simplified schema.
