"""
Week 3, Day 1 — Four Chunkers on One Document
=============================================

chunk_cost_sweep.py models chunking as geometry: uniformly-placed answer spans,
uniform chunk boundaries. Real documents are not uniform, and that difference is
the entire argument for structure-aware chunking.

This runs four strategies over the same policy excerpt and checks, for each of
three real questions, whether the answer survived intact:

    1. fixed         — hard cut every N tokens. The baseline nobody defends.
    2. recursive     — split on the largest separator that fits, backing off
                       through paragraph -> sentence -> word. The default in
                       every framework, and a genuinely good one.
    3. sentence      — pack whole sentences up to a budget. Never splits a
                       sentence; will happily split a clause from its exception.
    4. structural    — split on the document's own headings, then pack within a
                       section and stamp the section path onto every chunk.

The interesting result is not that structural wins. It is WHY it wins: in a
policy document the thing that makes a retrieved chunk answerable is the
heading it sat under, and that heading is 400 tokens above the text. Fixed and
recursive chunking discard it. That is not a tuning problem — no chunk size
recovers a fact that was never in the chunk.

No dependencies: tokens are approximated as whitespace-delimited words, which
is within ~30% of a real BPE count and does not change any conclusion here.

Usage:
    python chunker_compare.py
    python chunker_compare.py --size 120 --overlap 40
    python chunker_compare.py --show-chunks
"""

from __future__ import annotations

import argparse
import re

# ----------------------------------------------------------------------------
# A condensed but structurally realistic policy excerpt. Headings matter.
# ----------------------------------------------------------------------------

DOC = """\
# SECTION 4 — DEATH BENEFIT

## 4.1 Payment of Proceeds

Upon receipt of due proof of the death of the Insured while this Policy is in \
force, the Company will pay the Death Benefit to the named Beneficiary. Payment \
will be made within thirty days of receipt of all required documentation. \
Interest shall accrue on the proceeds from the date of death at the rate \
required by the law of the state in which this Policy was delivered.

## 4.2 Suicide Exclusion

If the Insured dies by suicide, while sane or insane, within two years from the \
Date of Issue, the Company's liability shall be limited to the return of \
premiums paid, without interest, less any indebtedness and less any partial \
withdrawals. This exclusion does not apply to any Policy issued in the State of \
Colorado, where the applicable period is one year from the Date of Issue.

## 4.3 Contestability

The Company will not contest this Policy after it has been in force during the \
lifetime of the Insured for two years from the Date of Issue, except for \
non-payment of premium and except for fraud where permitted by applicable state \
law. Reinstatement begins a new contestable period measured from the date of \
reinstatement, limited to statements made in the reinstatement application.

# SECTION 5 — RIDERS

## 5.1 Accidental Death Benefit Rider

The Company will pay an additional benefit equal to the Rider Face Amount if \
the Insured's death results directly from accidental bodily injury, independent \
of all other causes, and occurs within ninety days of the injury. No benefit is \
payable if death results from suicide, self-inflicted injury, or any cause \
excluded under Section 4.2 of the Policy.

## 5.2 Waiver of Premium Rider

If the Insured becomes totally disabled before age sixty and remains so for six \
consecutive months, the Company will waive the premiums falling due during the \
continuance of such disability. Written notice of claim must be given within \
one year of the commencement of disability.
"""

# The questions a user actually asks, and the text that must stay together for
# the retrieved chunk to answer them correctly.
QUESTIONS = [
    dict(q="Does the suicide exclusion apply in Colorado?",
         needs=["suicide", "Colorado", "one year"],
         why="The Colorado carve-out is the LAST sentence of 4.2. A chunk that "
             "holds the exclusion but not the carve-out produces a confidently "
             "wrong answer — the worst possible failure in this domain."),
    dict(q="Is accidental death covered if the death was a suicide?",
         needs=["Accidental Death", "suicide", "Section 4.2"],
         why="The rider answers by CROSS-REFERENCE. The chunk must carry both "
             "the rider heading and the pointer to 4.2, or retrieval lands on "
             "text that looks complete and is not."),
    dict(q="How long is the contestability period after reinstatement?",
         needs=["contest", "reinstatement", "new contestable period"],
         why="The general rule and its reinstatement exception are two "
             "sentences apart. Split them and you answer with the general rule."),
]


def toks(s: str) -> list[str]:
    return s.split()


def detok(t: list[str]) -> str:
    return " ".join(t)


# ----------------------------------------------------------------------------
# 1. Fixed
# ----------------------------------------------------------------------------

def chunk_fixed(doc: str, size: int, overlap: int) -> list[dict]:
    t = toks(doc)
    stride = max(1, size - overlap)
    out = []
    for i in range(0, len(t), stride):
        piece = t[i:i + size]
        if piece:
            out.append(dict(text=detok(piece), header=""))
        if i + size >= len(t):
            break
    return out


# ----------------------------------------------------------------------------
# 2. Recursive character/token splitting
# ----------------------------------------------------------------------------

SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_recursive(doc: str, size: int, overlap: int) -> list[dict]:
    def split(text: str, seps: list[str]) -> list[str]:
        if len(toks(text)) <= size:
            return [text]
        if not seps:
            t = toks(text)
            return [detok(t[i:i + size]) for i in range(0, len(t), size)]
        sep, rest = seps[0], seps[1:]
        parts, buf, out = text.split(sep), [], []
        for p in parts:
            trial = buf + [p]
            if len(toks(sep.join(trial))) <= size:
                buf = trial
            else:
                if buf:
                    out.append(sep.join(buf))
                buf = [p] if len(toks(p)) <= size else []
                if not buf:
                    out.extend(split(p, rest))
        if buf:
            out.append(sep.join(buf))
        return [o for o in out if o.strip()]

    pieces = split(doc, SEPARATORS)
    out = []
    for i, p in enumerate(pieces):
        # Framework-standard overlap: carry the tail of the previous piece.
        if i and overlap:
            tail = toks(pieces[i - 1])[-overlap:]
            p = detok(tail) + " " + p
        out.append(dict(text=p, header=""))
    return out


# ----------------------------------------------------------------------------
# 3. Sentence packing
# ----------------------------------------------------------------------------

def chunk_sentence(doc: str, size: int, overlap: int) -> list[dict]:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc) if s.strip()]
    out, buf = [], []
    for s in sents:
        if len(toks(" ".join(buf + [s]))) > size and buf:
            out.append(dict(text=" ".join(buf), header=""))
            # sentence-level overlap: keep trailing sentences within budget
            back, kept = 0, []
            for prev in reversed(buf):
                if back + len(toks(prev)) > overlap:
                    break
                kept.insert(0, prev)
                back += len(toks(prev))
            buf = kept
        buf.append(s)
    if buf:
        out.append(dict(text=" ".join(buf), header=""))
    return out


# ----------------------------------------------------------------------------
# 4. Structure-aware
# ----------------------------------------------------------------------------

def chunk_structural(doc: str, size: int, overlap: int) -> list[dict]:
    """Split on headings, pack within a section, stamp the section path."""
    out, h1, h2, buf = [], "", "", []

    def flush():
        nonlocal buf
        if not buf:
            return
        path = " > ".join(x for x in (h1, h2) if x)
        body = " ".join(buf)
        # Sections longer than the budget are packed by sentence, and EVERY
        # piece keeps the heading. That is the whole point.
        for piece in chunk_sentence(body, size, overlap):
            out.append(dict(text=f"[{path}] {piece['text']}", header=path))
        buf = []

    for line in doc.splitlines():
        if line.startswith("# "):
            flush()
            h1, h2 = line[2:].strip(), ""
        elif line.startswith("## "):
            flush()
            h2 = line[3:].strip()
        elif line.strip():
            buf.append(line.strip())
    flush()
    return out


STRATEGIES = [
    ("fixed", chunk_fixed),
    ("recursive", chunk_recursive),
    ("sentence", chunk_sentence),
    ("structural", chunk_structural),
]


# ----------------------------------------------------------------------------

def answerable(chunks: list[dict], needs: list[str]) -> tuple[bool, int]:
    """Does ANY single chunk contain every required fragment?"""
    hits = 0
    for c in chunks:
        low = c["text"].lower()
        if all(n.lower() in low for n in needs):
            hits += 1
    return hits > 0, hits


def report(args: argparse.Namespace) -> None:
    print("=" * 88)
    print("  FOUR CHUNKERS, ONE POLICY DOCUMENT")
    print("=" * 88)
    print(f"  Budget          : {args.size} tokens, {args.overlap} overlap"
          f" ({args.overlap / args.size:.0%})")
    print(f"  Document        : {len(toks(DOC)):,} tokens, 2 sections, 5 subsections")
    print()

    results = {}
    for name, fn in STRATEGIES:
        chunks = fn(DOC, args.size, args.overlap)
        results[name] = chunks

    print("-" * 88)
    print(f"  {'Strategy':<14}{'Chunks':>8}{'Avg tok':>10}{'Max tok':>10}"
          f"{'Keeps heading':>16}")
    print("  " + "-" * 84)
    for name, _ in STRATEGIES:
        ch = results[name]
        lens = [len(toks(c["text"])) for c in ch]
        keeps = "yes" if any(c["header"] for c in ch) else "NO"
        print(f"  {name:<14}{len(ch):>8}{sum(lens) / len(lens):>10.0f}"
              f"{max(lens):>10}{keeps:>16}")
    print()

    print("=" * 88)
    print("  CAN ANY SINGLE CHUNK ANSWER THE QUESTION?")
    print("=" * 88)
    score = {name: 0 for name, _ in STRATEGIES}
    for qi, q in enumerate(QUESTIONS, 1):
        print(f"  Q{qi}. {q['q']}")
        print(f"      needs together: {', '.join(q['needs'])}")
        for name, _ in STRATEGIES:
            ok, hits = answerable(results[name], q["needs"])
            score[name] += 1 if ok else 0
            mark = f"YES ({hits} chunk{'s' if hits != 1 else ''})" if ok else "NO"
            print(f"      {name:<14}{mark}")
        print(f"      why it is hard: ", end="")
        line, indent = "", " " * 22
        for w in q["why"].split():
            if len(line) + len(w) + 1 > 62:
                print(line if not line.startswith(" ") else line)
                print(indent, end="")
                line = w
            else:
                line = f"{line} {w}".strip()
        print(line)
        print()

    print("=" * 88)
    print(f"  SCORE AT {args.size} TOKENS")
    print("=" * 88)
    for name, _ in STRATEGIES:
        bar = "#" * (score[name] * 8)
        print(f"  {name:<14}{score[name]}/{len(QUESTIONS)}  {bar}")
    print()

    # ---- the informative version: how much budget does each strategy NEED? ---
    print("=" * 88)
    print("  ROBUSTNESS — SCORE vs CHUNK BUDGET")
    print("=" * 88)
    print("  A single budget proves nothing; pick a generous one and everything")
    print("  passes. What matters is how small a chunk each strategy survives.")
    print()
    budgets = [30, 40, 50, 60, 70, 90, 120]
    print(f"  {'Strategy':<14}" + "".join(f"{b:>8}" for b in budgets))
    print("  " + "-" * (14 + 8 * len(budgets)))
    survives = {}
    for name, fn in STRATEGIES:
        cells = []
        for b in budgets:
            ch = fn(DOC, b, max(1, b // 5))
            s = sum(1 for q in QUESTIONS if answerable(ch, q["needs"])[0])
            cells.append(s)
        survives[name] = min((b for b, s in zip(budgets, cells)
                              if s == len(QUESTIONS)), default=None)
        print(f"  {name:<14}" + "".join(f"{c}/{len(QUESTIONS):<6}" for c in cells))
    print()
    for name, _ in STRATEGIES:
        need = survives[name]
        txt = f"{need} tokens" if need else "never, at any budget tested"
        print(f"  {name:<14}first answers all three at: {txt}")
    print()
    print("  Read the FIXED row again — it scores 3/3 at 90 and drops back to 2/3 at")
    print("  120. Quality is not monotonic in chunk size for a chunker that ignores")
    print("  the document: a bigger chunk moves every boundary, and an alignment that")
    print("  happened to work stops working. That is not a tuning curve you can climb,")
    print("  it is luck, and it is why 'we tuned chunk size on our eval set' can mean")
    print("  'we overfit to which sentences happened to land together'.")
    print()
    print("  Structural holds at every budget because the heading travels WITH the")
    print("  text. The others need a chunk large enough to physically span from the")
    print("  heading down to the clause — which means the budget is set by the")
    print("  document's layout, not by what makes a good retrieval unit.")
    print()

    print("  WHY STRUCTURAL WINS, AND WHAT IT IS NOT")
    for line in [
        "It is not that the boundaries are better placed. It is that every chunk",
        "carries its section path, so 'suicide, self-inflicted injury, or any cause",
        "excluded under Section 4.2' arrives labelled ACCIDENTAL DEATH BENEFIT RIDER",
        "instead of arriving as an anonymous paragraph about suicide. The embedding",
        "changes, the retrieval changes, and the model can tell which document it is",
        "reading.",
        "",
        "This is also the cheapest possible version of 'contextual retrieval': you",
        "are prepending context the document already gave you, for free, at ingest.",
        "The expensive version — an LLM writing a summary for every chunk — buys",
        "more, but start here, because a document with headings has already done",
        "most of the work and you are throwing it away.",
    ]:
        print(f"  {line}")
    print()

    print("  THE HONEST COUNTERPOINT")
    for line in [
        "Structural chunking needs structure. It is excellent on filed policy forms,",
        "regulatory text, and API docs; it does nothing for a support-ticket thread,",
        "a call transcript, or scraped product copy. Recursive splitting with good",
        "separators is the right default precisely BECAUSE it degrades gracefully",
        "when the document has no skeleton to follow.",
    ]:
        print(f"  {line}")

    if args.show_chunks:
        print()
        print("=" * 88)
        print("  CHUNKS")
        print("=" * 88)
        for name, _ in STRATEGIES:
            print(f"\n  --- {name} ---")
            for i, c in enumerate(results[name]):
                body = c["text"][:150].replace("\n", " ")
                print(f"  [{i:>2}] {body}...")
    print("=" * 88)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=90, help="chunk budget in tokens")
    p.add_argument("--overlap", type=int, default=20)
    p.add_argument("--show-chunks", action="store_true")
    report(p.parse_args())


if __name__ == "__main__":
    main()
