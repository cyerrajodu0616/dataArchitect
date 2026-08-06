"""
Week 3, Day 2 — BM25 From Scratch, and the IDF Postgres Does Not Give You
=========================================================================

Week 2 Day 4 noted in passing that `ts_rank_cd` has no IDF. This script works
out what that actually costs, and the measured answer is not the one you would
guess: **the trigger is QUERY LENGTH, not corpus size.**

Two scorers over identical term statistics:

    bm25    — real Okapi BM25: TF saturation (k1), length normalisation (b),
              and inverse document frequency.
    no_idf  — the same machinery with IDF removed. This isolates the single
              property `ts_rank_cd` lacks.

`ts_rank_cd` is *cover density* ranking. It rewards query terms appearing close
together and respects the A/B/C/D weight labels you assign with setweight().
It is genuinely good at proximity and this is not an argument that it is a bad
function. It simply does not know that "reinstatement" is rarer than "period",
and this script measures what that is worth.

WHAT IT SHOWS

  Part 1  IDF over a small realistic corpus — the table that IS the argument.

  Part 2  The same corpus scored both ways, where the two scorers AGREE. This
          is the honest control: on a curated five-document corpus the answer
          contains the rare term and the common ones, so it wins either way.
          Anyone evaluating ts_rank_cd on a sample corpus concludes it is fine,
          because at that scale it is.

  Part 3  A controlled experiment that adds what a curated corpus lacks —
          DECOY documents that repeat the common query terms without containing
          the answer. Sweeping corpus size against query length gives a clean
          result:

            BM25 holds rank 1 in every cell.
            The no-IDF scorer loses the answer from 6 query terms onward,
            at EVERY corpus size tested.

          So the threshold is set by query length. Corpus size sets the
          MAGNITUDE — the IDF spread between a rare and a common term runs from
          ~4.8x at five documents to ~66x at four million — which is why the
          same defect is a rounding error in development and most of your
          ranking quality in production.

That maps directly onto Week 2 Day 4's query classes: the NATURAL_LANGUAGE
class is precisely the one this breaks, and it is the class where teams assume
the dense arm is carrying the query anyway.

stdlib only.

Usage:
    python bm25_vs_tsrank.py
    python bm25_vs_tsrank.py --domain fintech
    python bm25_vs_tsrank.py --domain retail --k1 2.0 --b 0.3
    python bm25_vs_tsrank.py --experiment-only
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

# ----------------------------------------------------------------------------
# Small realistic corpora, for the IDF table.
# ----------------------------------------------------------------------------

CORPORA = {
    "insurance": dict(
        docs=[
            "The grace period for premium payment is thirty one days from the due "
            "date. During the grace period the policy remains in force. If premium "
            "is not paid by the end of the grace period the policy will lapse.",
            "The Company will not contest this policy after it has been in force "
            "during the lifetime of the insured for two years from the date of "
            "issue. Reinstatement begins a new contestable period measured from "
            "the date of reinstatement.",
            "The free look period allows the owner to return the policy within ten "
            "days of delivery for a full refund of premium paid. This period may "
            "be longer in some states.",
            "Policy values accumulate at the guaranteed interest rate during the "
            "accumulation period. The surrender period runs for ten policy years.",
            "A policy that has lapsed may be reinstated within three years subject "
            "to evidence of insurability and payment of all overdue premium.",
        ],
        query="contestable period after reinstatement",
        rel=1,
    ),
    "fintech": dict(
        docs=[
            "A chargeback is initiated by the issuing bank on behalf of the "
            "cardholder. The merchant may respond with compelling evidence.",
            "Reason code 4853 covers cardholder disputes where the goods or "
            "services were not as described. The merchant must provide proof of "
            "delivery matching the order.",
            "Dispute resolution timelines vary by network. The merchant has forty "
            "five days to respond to a dispute.",
            "Reason code 4837 covers no cardholder authorisation. This is a fraud "
            "reason code and the merchant generally cannot represent it.",
            "Merchants should monitor their chargeback ratio. A chargeback ratio "
            "above one percent places the merchant in a monitoring programme.",
        ],
        query="merchant chargeback reason code 4853",
        rel=1,
    ),
    "retail": dict(
        docs=[
            "This laptop sleeve fits most 13 inch and 14 inch laptops. The sleeve "
            "is water resistant with a soft interior lining.",
            "USB C to HDMI adapter supporting 4K at 60Hz. Compatible with "
            "Thunderbolt 3 and Thunderbolt 4 ports.",
            "Replacement charger rated at 65W with USB C power delivery. "
            "Compatible with most USB C laptops.",
            "Laptop stand with adjustable height. The stand supports laptops up "
            "to 17 inches and folds flat for travel.",
            "Screen protector for 13 inch laptop displays. Reduces glare and "
            "resists fingerprints.",
        ],
        query="laptop thunderbolt adapter",
        rel=1,
    ),
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Index:
    """Term statistics plus both scorers. Identical inputs, one difference."""

    def __init__(self, docs: list[str]):
        self.docs = docs
        self.toks = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.lens = [len(t) for t in self.toks]
        self.avgdl = sum(self.lens) / len(self.lens)
        self.N = len(docs)
        self.df: Counter = Counter()
        for t in self.toks:
            for term in set(t):
                self.df[term] += 1

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def _score(self, query: str, k1: float, b: float, use_idf: bool) -> list[float]:
        out = []
        for i in range(self.N):
            s = 0.0
            for term in tokenize(query):
                f = self.tf[i].get(term, 0)
                if not f:
                    continue
                denom = f + k1 * (1 - b + b * self.lens[i] / self.avgdl)
                w = self.idf(term) if use_idf else 1.0
                s += w * f * (k1 + 1) / denom
            out.append(s)
        return out

    def bm25(self, q: str, k1: float, b: float) -> list[float]:
        return self._score(q, k1, b, True)

    def no_idf(self, q: str, k1: float, b: float) -> list[float]:
        return self._score(q, k1, b, False)


def ranks(scores: list[float]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    r = [0] * len(scores)
    for pos, i in enumerate(order, 1):
        r[i] = pos
    return r


# ----------------------------------------------------------------------------
# Part 3 — the controlled experiment
# ----------------------------------------------------------------------------

COMMON = ["policy", "period", "coverage", "payment", "insured", "benefit",
          "premium", "company", "amount", "date"]
RARE = "contestable"
FILLER = ["notice", "office", "written", "owner", "change", "request", "form",
          "value", "rate", "term", "issue", "effect", "state", "law", "may"]


def build_corpus(n_docs: int, n_common: int, n_decoys: int) -> tuple[list[str], int]:
    """
    Deterministic corpus.

      doc 0        the ANSWER: every query term exactly once, including the rare
                   discriminator.
      docs 1..D    DECOYS: every COMMON query term twice, the rare term never.
      rest         BACKGROUND: every COMMON term once — this is what MAKES them
                   common — plus rotating filler, and never the rare term.

    The background documents are the important detail. Without them the
    "common" terms appear in only a handful of documents, their IDF is high, and
    the experiment measures nothing. Term rarity is a property of the CORPUS,
    so the corpus has to be built to have it.

    Lengths are held close so length normalisation is not the variable.
    """
    common = COMMON[:n_common]
    pad = FILLER[:8]
    docs = []

    docs.append(" ".join(common + [RARE] + pad))

    decoy = " ".join(common + common + pad)
    for _ in range(n_decoys):
        docs.append(decoy)

    while len(docs) < n_docs:
        i = len(docs)
        rot = FILLER[i % len(FILLER):] + FILLER[: i % len(FILLER)]
        docs.append(" ".join(common + rot[:9]))

    return docs[:n_docs], 0


def experiment(args: argparse.Namespace) -> None:
    print("=" * 90)
    print("  PART 3 — WHEN DOES THE MISSING IDF ACTUALLY CHANGE THE RANKING?")
    print("=" * 90)
    print("  Setup: one ANSWER document containing every query term once; decoy")
    print("  documents containing every COMMON term twice and the rare term never.")
    print("  Lengths held comparable so length normalisation is not the variable.")
    print()
    print("  Reported: the rank of the ANSWER document under each scorer.")
    print("  Rank 1 = correct. Anything else means the decoys won.")
    print()

    sizes = [5, 20, 100, 1_000, 10_000]
    lengths = [2, 4, 6, 8, 10]

    print(f"  {'query terms':<14}" + "".join(f"{('N=' + f'{s:,}'):>15}" for s in sizes))
    print("  " + "-" * (14 + 15 * len(sizes)))

    first_break = None
    for nq in lengths:
        n_common = nq - 1
        cells = []
        for n in sizes:
            docs, rel = build_corpus(n, n_common, args.decoys)
            ix = Index(docs)
            q = " ".join(COMMON[:n_common] + [RARE])
            br = ranks(ix.bm25(q, args.k1, args.b))[rel]
            nr = ranks(ix.no_idf(q, args.k1, args.b))[rel]
            cells.append((br, nr))
            if nr > 1 and first_break is None:
                first_break = (nq, n)
        row = "".join(f"{f'{b} / {n}':>15}" for b, n in cells)
        print(f"  {nq:<14}{row}")

    print("  " + "-" * (14 + 15 * len(sizes)))
    print("  each cell is  BM25 rank / no-IDF rank")
    print()

    if first_break:
        nq, _ = first_break
        for line in [
            "BM25 holds rank 1 in every cell. The no-IDF scorer loses the answer",
            f"from {nq} query terms onward, at EVERY corpus size.",
            "",
            "Read that carefully, because it is not the result you might expect:",
            "the driver here is QUERY LENGTH, not corpus size. Once the query",
            "carries enough common words, a decoy that repeats them all outscores",
            "the single document containing the discriminator — and saturation",
            "means the answer's one rare term cannot make up the difference",
            "without an IDF weight in front of it.",
            "",
            "Corpus size changes the MAGNITUDE, not the threshold. See below.",
        ]:
            print(f"  {line}")
    else:
        print("  No divergence at these settings — raise --decoys or query length.")
    print()

    # --- why: the IDF spread grows with N -----------------------------------
    print("-" * 90)
    print("  WHY IT GETS WORSE WITH SCALE: IDF SPREAD vs CORPUS SIZE")
    print("-" * 90)
    print(f"  {'corpus':>10}{'IDF(rare, in 1 doc)':>24}{'IDF(common, in 80%)':>24}"
          f"{'ratio':>12}")
    print("  " + "-" * 86)
    for n in [5, 100, 10_000, 1_000_000, 4_000_000]:
        idf_rare = math.log(1 + (n - 1 + 0.5) / 1.5)
        nc = 0.8 * n
        idf_common = math.log(1 + (n - nc + 0.5) / (nc + 0.5))
        print(f"  {n:>10,}{idf_rare:>24.3f}{idf_common:>24.3f}"
              f"{idf_rare / idf_common:>12.1f}x")
    print()
    for line in [
        "A rare term's discriminating power grows as ln(N); a common term's decays",
        "toward zero. The spread runs from ~4.8x at five documents to ~66x at four",
        "million. So IDF is not a tie-breaker at production scale — it is most of",
        "what ranking IS, and a scorer without it is optimising something else.",
        "",
        "Why the gap survives into production, precisely:",
        "",
        "  * A development corpus is curated. It has few DECOYS — documents that",
        "    repeat your common query terms without containing the answer. Part 2",
        "    is exactly that corpus, and both scorers agree on it.",
        "  * A production corpus is full of them, because most documents are about",
        "    roughly the same subject and share the same vocabulary.",
        "  * And short keyword queries stay safe throughout, so early manual",
        "    testing — which is always short queries — looks fine.",
        "",
        "The failure therefore appears when real users type sentences at a real",
        "corpus, which is after launch.",
    ]:
        print(f"  {line}")


def report(args: argparse.Namespace) -> None:
    data = CORPORA[args.domain]
    ix = Index(data["docs"])
    q = data["query"]
    rel = data["rel"]

    print("=" * 90)
    print("  BM25 vs THE SAME SCORER WITHOUT IDF")
    print("=" * 90)
    print(f"  Domain     : {args.domain.upper()}")
    print(f"  Corpus     : {ix.N} documents, avg length {ix.avgdl:.1f} tokens")
    print(f"  Parameters : k1={args.k1} (TF saturation), b={args.b} (length norm)")
    print(f"  Query      : \"{q}\"")
    print()

    print("-" * 90)
    print("  PART 1 — IDF OF EVERY QUERY TERM")
    print("-" * 90)
    print(f"  {'term':<18}{'in docs':>10}{'IDF':>10}   contribution")
    print("  " + "-" * 86)
    for t in tokenize(q):
        n = ix.df.get(t, 0)
        bar = "#" * int(ix.idf(t) * 10)
        print(f"  {t:<18}{n:>6} / {ix.N:<3}{ix.idf(t):>10.3f}   {bar}")
    print()
    print("  A term in every document has IDF near zero and contributes nothing to")
    print("  the ranking. A term in one document dominates it. Delete IDF and that")
    print("  ordering collapses into 'whoever repeated the words most'.")
    print()

    b_scores = ix.bm25(q, args.k1, args.b)
    n_scores = ix.no_idf(q, args.k1, args.b)
    b_rank, n_rank = ranks(b_scores), ranks(n_scores)

    print("-" * 90)
    print("  PART 2 — AND YET, ON FIVE DOCUMENTS, IT BARELY MATTERS")
    print("-" * 90)
    print(f"  {'#':>3}  {'BM25':>8}{'rank':>6}   {'no-IDF':>8}{'rank':>6}   document")
    print("  " + "-" * 86)
    for i in range(ix.N):
        mark = " <-- ANSWERS IT" if i == rel else ""
        print(f"  {i:>3}  {b_scores[i]:>8.3f}{b_rank[i]:>6}   "
              f"{n_scores[i]:>8.3f}{n_rank[i]:>6}   "
              f"{data['docs'][i][:30]}...{mark}")
    print()
    verdict = ("They agree." if b_rank[rel] == n_rank[rel]
               else f"They disagree: BM25 {b_rank[rel]}, no-IDF {n_rank[rel]}.")
    print(f"  Correct document is #{rel}. {verdict}")
    print()
    for line in [
        "This is the honest result, and it is the point. On a five-document corpus",
        "the answer contains the rare term AND the common ones, so it wins under",
        "either scorer. Length normalisation helps it further.",
        "",
        "Anyone who evaluates ts_rank_cd on a sample corpus will conclude it is",
        "fine, because at this scale it IS fine. Part 3 is where it stops being.",
    ]:
        print(f"  {line}")
    print()


def guidance() -> None:
    print("=" * 90)
    print("  WHAT TO DO ABOUT IT IN POSTGRES")
    print("=" * 90)
    for line in [
        "In the order worth trying:",
        "",
        "  1. WEIGHT THE FIELDS. setweight() gives title/SKU/code columns label A",
        "     and body label D. This does not restore IDF, but it approximates it",
        "     for the case that matters most: discriminators usually live in",
        "     specific columns. Cheapest fix by a wide margin, pure SQL, no new",
        "     moving parts.",
        "",
        "        setweight(to_tsvector('english', coalesce(sku, '')),      'A') ||",
        "        setweight(to_tsvector('english', coalesce(title, '')),    'B') ||",
        "        setweight(to_tsvector('english', coalesce(body, '')),     'D')",
        "",
        "  2. COMPUTE IDF YOURSELF. ts_stat() gives document frequencies. A",
        "     materialised term -> idf table plus a scoring expression gets you",
        "     real BM25 in SQL. It works; it also needs refreshing as the corpus",
        "     drifts, which is a job you now own.",
        "",
        "  3. LET THE DENSE ARM CARRY IT. If the rare term is semantically loaded",
        "     the embedding may rank it correctly anyway, and RRF lets the dense",
        "     side win. This is the honest reason hybrid search papers over the",
        "     problem — and the reason it is easy never to notice.",
        "",
        "Option 3 is why this is an architecture issue and not a tuning issue:",
        "your lexical arm may be weaker than you think and fusion is hiding it.",
        "Measure the arms SEPARATELY — Week 2 Day 4's per-class evaluation — or",
        "you will credit the wrong half of the system and tune the wrong knob.",
        "",
        "The class most at risk is NATURAL_LANGUAGE: long queries made mostly of",
        "common words plus one discriminator. The class least at risk is",
        "EXACT_ID, where the query is nothing but the rare term.",
    ]:
        print(f"  {line}")
    print("=" * 90)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(CORPORA), default="insurance")
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--decoys", type=int, default=3)
    p.add_argument("--experiment-only", action="store_true")
    args = p.parse_args()

    if not args.experiment_only:
        report(args)
    experiment(args)
    guidance()


if __name__ == "__main__":
    main()
