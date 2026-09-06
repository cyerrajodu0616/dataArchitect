# Deepen Week 4 Day 1 with a real LangGraph implementation

## Context

`Claude/week4/day1/0019-langgraph-fundamentals.html` explains state machines well, but its only runnable artifact is the custom `Graph` class in `state_machine_trace.py`. The lesson names LangGraph APIs without giving the learner an executable `StateGraph`. Expand Day 1 from conceptual understanding to implementation, execution tracing, limits, and tests while keeping the existing beginner ramp.

Official API verification performed against the LangGraph Graph API and streaming documentation on 2026-09-06. PyPI reports LangGraph 1.2.11 with Python >=3.10 support; this repository uses Python >=3.14.

## Exact changes

### `pyproject.toml`

At lines 8–14, add `langgraph>=1.2.11,<2.0.0` to the existing dependency array without changing other dependencies.

Before:

```toml
dependencies = [
    "matplotlib>=3.11.1",
    "numpy>=2.5.1",
    "openai>=2.52.0",
    "pandas>=3.0.5",
    "python-dotenv>=1.2.2",
    "scikit-learn>=1.9.0",
]
```

After: the same block with `"langgraph>=1.2.11,<2.0.0",` before `matplotlib`.

### `Claude/week4/day1/0019-langgraph-fundamentals.html`

1. At lines 246–249, retain the existing “Try this” block but replace its final absolute sentence with language that says the custom loop demonstrates LangGraph's execution core while the real runtime also defines merge semantics, validates topology, schedules super-steps, streams events, applies limits, and integrates persistence.
2. At lines 280–288, revise only the “retry” and “picks different tools” rows. Explain that a local transient retry belongs inside a node/retry policy; a graph edge is appropriate when another attempt changes shared state or routing. Explain that one model/tool loop can be plain code, while explicit multi-step routing benefits from a graph.
3. At lines 290–293, replace “count backward edges” as the sole test with a decision test based on durable pause/resume, cross-step loops, explicit inspectable routing, and recoverable state. State explicitly that conditional logic alone does not require a graph.
4. Insert a new section immediately before the existing line 327 `<h2>7. What LangGraph Adds On Top</h2>`. The section must be titled `7. Build the Same Agent in Real LangGraph` and cover:
   - the `TypedDict` state schema;
   - reducer semantics using `Annotated[list[str], operator.add]` versus scalar overwrite;
   - nodes returning partial state updates;
   - graph construction with `StateGraph`, `START`, `END`, `add_node`, `add_edge`, and `add_conditional_edges`;
   - `compile()` as validation/runtime construction rather than execution;
   - `invoke()` for final state and `stream(..., stream_mode="updates")` for per-node deltas;
   - `recursion_limit` as a runtime backstop, not a business exit condition;
   - a warning that checkpoint boundaries are super-step boundaries and a resumed node can rerun from its beginning, so side effects require idempotency;
   - a “production boundary” table distinguishing what the deterministic example teaches from what Days 3–5 add;
   - commands for running the example, streaming it, triggering the runaway guard, and running its tests.
5. Renumber the current sections 7, 8, and 9 to 8, 9, and 10.
6. Update the companion-script table to list both the existing custom engine and `langgraph_refund_agent.py`, plus `test_langgraph_refund_agent.py`.
7. Update the final ask-agent copy so it offers help tracing the real graph and designing state/reducers/exit conditions.

Insertion anchor copied from current lines 321–331:

```html
<div class="caution">
<div class="label">Notice what differs across the three</div>
The <em>shape</em> is identical in all three — a gather/check loop feeding a decision. What differs is the <strong>exit condition</strong>, and in each case it comes from the business, not from engineering: customer patience, an underwriting deadline, a regulatory clock.
<p style="margin-top:0.7rem;margin-bottom:0;">That's the thing to notice as an architect. The interesting design decisions in an agent are almost never "which framework" — they're "when does this stop, and who decided that."</p>
</div>

<h2>7. What LangGraph Adds On Top</h2>

<p>The 40-line engine in the companion script is a real state machine. LangGraph is that plus the things you'd otherwise write yourself:</p>
```

The insertion goes between `</div>` and the existing heading; the existing heading becomes `<h2>8. What LangGraph Adds On Top</h2>`.

### Create `Claude/week4/day1/langgraph_refund_agent.py`

Create a deterministic, API-key-free real LangGraph version of the retail agent with:

- `AgentState(TypedDict, total=False)` containing `messages`, `order_id`, `intent`, `confident`, `clarify_count`, `clarification_replies`, `order`, `within_window`, `days_since_delivery`, `decision`, and `reason`;
- `messages: Annotated[list[str], operator.add]` so node-returned messages append;
- the existing `CLEAR_INTENTS` and `ORDERS` values copied from `state_machine_trace.py` lines 146–155;
- nodes `classify`, `ask_clarify`, `lookup_order`, `check_policy`, `approve`, `escalate`, `handle_track`, `handle_cancel`, and `give_up`, each returning only changed fields;
- routing functions `after_classify` and `after_policy` with explicit `Literal` return types;
- `build_graph(runaway=False)` using the public Graph API and returning a compiled graph;
- `initial_state()` and `run_agent()` public helpers;
- CLI options `--request`, `--order-id`, `--stream`, `--runaway`, and `--max-steps`;
- streaming output that prints the node name and update from `stream_mode="updates"`;
- `GraphRecursionError` handling that clearly identifies the runtime guard;
- no external model call, network call, checkpointer, or hidden global mutation.

The script must state that the deterministic classifier isolates orchestration mechanics; replacing the classifier with a model is a later concern.

### Create `Claude/week4/day1/test_langgraph_refund_agent.py`

Use standard-library `unittest` to cover:

1. ambiguous request plus default clarification becomes `SENT_TRACKING`;
2. the message reducer preserves the original message and appends the clarification;
3. a refund inside the return window is approved;
4. a refund outside the window is escalated;
5. two unresolved clarification replies reach `HANDED_TO_AGENT`;
6. streamed updates include `classify`, `ask_clarify`, and `handle_track` in order;
7. a runaway graph raises `GraphRecursionError` under a small runtime recursion limit.

The test must add its own directory to `sys.path` before importing the sibling companion module so the documented repository-root unittest command works as well as execution from the Day 1 directory.

## Verification

1. Update the lock file through the project's package manager.
2. Run `uv run python Claude/week4/day1/langgraph_refund_agent.py`.
3. Run it with `--stream`, a clear refund request for each order, and `--runaway --max-steps 4`.
4. Run `uv run python -m unittest Claude/week4/day1/test_langgraph_refund_agent.py -v`.
5. Run the existing `state_machine_trace.py` commands to prove no regression.
6. Parse the modified HTML and verify headings, links, quiz script, and balanced structural tags.
7. Run `git diff --check`.

## Constraints and gotchas

- Preserve the existing deterministic custom engine and lesson behavior.
- Do not require an OpenAI key or make network calls in the example or tests.
- A reducer applies to updates; scalar fields without reducers overwrite by default.
- The runtime `recursion_limit` is a safety guard and must not be described as the loop's business exit condition.
- LangGraph checkpoints occur at super-step boundaries; do not imply arbitrary line-level resume.
- Preserve unrelated untracked interview-profile files currently in the worktree.
- Do not touch deployment or workflow configuration.
