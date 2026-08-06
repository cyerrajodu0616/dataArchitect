"""
Week 2, Day 4 — Retail Query Classifier & Hybrid Router
=======================================================

The classifier box from the lesson's pipeline diagram. Given a query, decide:

    - which class it belongs to,
    - whether to SHORT-CIRCUIT to an exact lookup (no fusion at all),
    - and if not, what RRF weights the lexical and dense arms should get.

The whole point is that a single global alpha optimises the *average* query,
while retail traffic is five populations with different needs and very
different conversion value. Routing costs microseconds of regex and beats a
tuned global weight comfortably.

Deliberately heuristic — regex and token statistics, no model call. Start here.
Only reach for an LLM classifier if this plateaus, and measure before you do.

stdlib only.

Usage:
    python query_class_router.py
    python query_class_router.py --query "Salomon Quest 4 GTX size 11"
"""

from __future__ import annotations

import argparse
import difflib
import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Domain knowledge. In production these come from your catalog, refreshed
# nightly — brands from the brand dimension, vocabulary from the tsvector
# lexeme table, attributes from your facet configuration.
# ----------------------------------------------------------------------------

BRANDS = {
    "salomon", "dyson", "patagonia", "merrell", "columbia", "northface",
    "arcteryx", "keen", "hoka", "garmin", "weber", "lodge", "yeti",
}

ATTRIBUTE_TERMS = {
    "waterproof", "insulated", "lightweight", "wide", "narrow", "slim",
    "leather", "fleece", "goretex", "breathable", "cordless", "stainless",
    "black", "brown", "navy", "olive", "red", "grey", "gray",
    "mens", "womens", "kids", "size", "inch", "litre", "liter", "cm", "mm",
}

# A stand-in for the catalog vocabulary used to detect misspellings.
VOCABULARY = {
    "waterproof", "hiking", "boots", "boot", "jacket", "trail", "running",
    "shoes", "backpack", "tent", "sleeping", "bag", "vacuum", "cleaner",
    "cordless", "insulated", "fleece", "gloves", "socks", "trousers",
    "warm", "rain", "winter", "summer", "lightweight", "wide",
} | BRANDS | ATTRIBUTE_TERMS

FUNCTION_WORDS = {
    "for", "with", "a", "an", "the", "something", "that", "to", "in", "on",
    "my", "me", "i", "need", "want", "looking", "good", "best", "and", "of",
}

# Identifier shapes. Real catalogs need these tuned to your own SKU grammar.
IDENTIFIER_PATTERNS = [
    re.compile(r"^[A-Z]{2,4}[-_#]?\d{4,}[-A-Z0-9]*$", re.I),  # SKU-88213
    re.compile(r"^(mfg|item|part|model)\s*#?\s*[\w-]{4,}$", re.I),  # MFG#4471-B
    re.compile(r"^\d{8,14}$"),  # bare UPC/EAN
]


@dataclass
class Route:
    query: str
    query_class: str
    short_circuit: bool
    w_lexical: float
    w_dense: float
    signals: list[str] = field(default_factory=list)
    strategy: str = "weighted RRF (k=60)"
    note: str = ""


def _tokens(q: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9#-]+", q.lower())


def _looks_like_identifier(token: str) -> bool:
    if any(p.match(token) for p in IDENTIFIER_PATTERNS):
        return True
    # Mixed alphanumeric with a run of 4+ digits: strong identifier signal.
    return bool(re.search(r"\d{4,}", token)) and bool(re.search(r"[a-z]", token, re.I))


ACQUIRER_PREFIXES = {"sq", "tst", "sp", "pp", "paypal", "amzn", "wl", "iz", "chk"}


def _looks_like_merchant_descriptor(q: str) -> bool:
    """
    Raw card-network merchant descriptors: "AMZN MKTP US*2H4TR", "SQ *BLUE
    BOTTLE COFFEE", "TST* THE LOCAL CAFE".

    Signals: uppercase-dominant, an acquirer prefix and/or a '*' separator, and
    no natural-language function words. Deliberately conservative — a false
    positive here sends a real question down the merchant path.
    """
    alpha = [c for c in q if c.isalpha()]
    if len(alpha) < 3:
        return False
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    toks = re.findall(r"[A-Za-z0-9]+", q.lower())
    if any(t in FUNCTION_WORDS for t in toks):
        return False
    has_star = "*" in q
    has_prefix = bool(toks) and toks[0] in ACQUIRER_PREFIXES
    return upper_ratio > 0.85 and (has_star or has_prefix)


def _misspelled(tokens: list[str]) -> list[str]:
    """Tokens that are near-misses against catalog vocabulary."""
    out = []
    for t in tokens:
        if len(t) < 4 or t in VOCABULARY or t in FUNCTION_WORDS or t.isdigit():
            continue
        if _looks_like_identifier(t):
            continue
        close = difflib.get_close_matches(t, VOCABULARY, n=1, cutoff=0.75)
        if close:
            out.append(f"{t}->{close[0]}")
    return out


def classify(query: str, domain: str = "retail") -> Route:
    q = query.strip()
    tokens = _tokens(q)
    signals: list[str] = []

    # --- 0. Domain-specific identifier grammars (checked first) -----------
    for pat, kind in DOMAIN_PATTERNS.get(domain, []):
        if pat.match(q):
            signals.append(f"matches {domain} pattern: {kind}")
            if kind == "ICD-10 diagnosis code":
                return Route(
                    q, "MEDICAL_CODE", True, 1.0, 0.0, signals,
                    strategy="exact lookup on a code table — NEVER fuse, NEVER nearest-neighbour",
                    note="E11.9 (diabetes without complications) and E11.29 (with kidney "
                         "complications) embed almost identically but can mean accept-at-"
                         "standard versus decline. Dense retrieval cannot distinguish them "
                         "and will return the wrong one confidently. This class must be an "
                         "exact code lookup; if the code is unknown, fail loudly rather "
                         "than returning a near match.",
                )
            return Route(
                q, "EXACT_IDENTIFIER", True, 1.0, 0.0, signals,
                strategy=f"btree lookup on the normalised {kind} column — bypass fusion",
                note="In a regulated domain, returning a near-miss identifier is a "
                     "compliance event rather than a conversion loss. Short-circuit, and "
                     "if the lookup misses, return nothing rather than something close.",
            )

    # --- 0b. Fintech merchant descriptor strings --------------------------
    # The canonical merchant-normalisation problem: raw card-network descriptors
    # like "AMZN MKTP US*2H4TR" or "SQ *BLUE BOTTLE COFFEE". Neither arm alone
    # works — lexical anchors the recognisable brand token, dense generalises
    # across the acquirer noise around it. Dense-dominant is actively wrong here.
    if domain == "fintech" and _looks_like_merchant_descriptor(q):
        signals.append("card-network merchant descriptor "
                       "(uppercase-dominant, acquirer prefix and/or '*' separator)")
        return Route(
            q, "MERCHANT_STRING", False, 0.65, 0.35, signals,
            strategy="weighted RRF + pg_trgm fuzzy arm against the merchant catalog",
            note="Lexical-dominant: the brand token ('AMZN', 'BLUE BOTTLE') is the "
                 "high-signal part, while the acquirer prefix and terminal suffix are "
                 "noise that dense retrieval will happily match on. Add a trigram arm "
                 "because these strings are truncated and abbreviated inconsistently "
                 "across acquirers.",
        )

    # --- 1. Exact identifier: short-circuit, do not fuse -------------------
    whole = q.replace(" ", "")
    id_tokens = [t for t in tokens if _looks_like_identifier(t)]
    if any(p.match(q) for p in IDENTIFIER_PATTERNS) or any(
        p.match(whole) for p in IDENTIFIER_PATTERNS
    ):
        signals.append("query matches an identifier pattern end-to-end")
        return Route(
            q, "EXACT_IDENTIFIER", True, 1.0, 0.0, signals,
            strategy="btree lookup on sku_norm — bypass fusion entirely",
            note="Highest-converting class. Blending dense results here can only "
                 "dilute a perfect match. If the lookup misses, fall through to "
                 "trigram fuzzy match before touching the dense arm.",
        )

    # --- 2. Brand + model -------------------------------------------------
    brand_hits = [t for t in tokens if t in BRANDS]
    if brand_hits:
        signals.append(f"brand token(s): {', '.join(brand_hits)}")
        if id_tokens or any(re.search(r"\d", t) for t in tokens):
            signals.append("model-number-like token present")
            return Route(
                q, "BRAND_MODEL", False, 0.75, 0.25, signals,
                note="Rare high-signal tokens: lexical-dominant. Dense retains a "
                     "minority weight to catch variants and near-misses.",
            )
        signals.append("brand without a model number")
        return Route(
            q, "BRAND_MODEL", False, 0.6, 0.4, signals,
            note="Brand is discriminative but the rest of the query is descriptive.",
        )

    # --- 3. Misspelled ----------------------------------------------------
    miss = _misspelled(tokens)
    if miss and len(miss) >= max(1, len(tokens) // 3):
        signals.append(f"near-miss tokens: {', '.join(miss)}")
        return Route(
            q, "MISSPELLED", False, 0.25, 0.75, signals,
            note="Dense is robust to typos via subword tokenisation; exact-match "
                 "lexical fails outright. Add a pg_trgm fuzzy arm rather than "
                 "relying on tsvector here.",
        )

    # --- 4. Attribute-constrained ----------------------------------------
    attr_hits = [t for t in tokens if t in ATTRIBUTE_TERMS]
    has_measure = bool(re.search(r"\b\d+(\.\d+)?\s?(inch|in|cm|mm|l|litre|liter)?\b", q, re.I))
    if len(attr_hits) >= 2 or (attr_hits and has_measure):
        signals.append(f"attribute term(s): {', '.join(attr_hits)}")
        if has_measure:
            signals.append("numeric measurement present -> extract as a SQL filter")
        return Route(
            q, "ATTRIBUTE_CONSTRAINED", False, 0.5, 0.5, signals,
            note="Balanced fusion, but the real win is extracting size/colour/spec "
                 "into a WHERE clause so both arms search a smaller, correct set.",
        )

    # --- 5. Natural language (default) -----------------------------------
    fw = [t for t in tokens if t in FUNCTION_WORDS]
    if fw:
        signals.append(f"function words present: {', '.join(fw)}")
    if len(tokens) >= 5:
        signals.append(f"long query ({len(tokens)} tokens)")
    if not signals:
        signals.append("no lexical or structural signal — treated as descriptive")
    return Route(
        q, "NATURAL_LANGUAGE", False, 0.25, 0.75, signals,
        note="Dense-dominant. BM25 contributes little and can actively hurt by "
             "matching incidental words in unrelated descriptions.",
    )


# ----------------------------------------------------------------------------
# Domain-specific identifier grammars.
#
# The taxonomy transfers across domains; the identifier SHAPES and the cost of
# getting them wrong do not. Insurance adds a class retail has no analogue for:
# medical codes, where ICD-10 E11.9 and E11.29 are different conditions with
# different underwriting outcomes, and whose embeddings are nearly identical.
# ----------------------------------------------------------------------------

DOMAIN_PATTERNS = {
    "insurance": [
        (re.compile(r"^[A-Z]{2,4}\d{2}-[A-Z]{1,4}(-[A-Z]{2})?$", re.I), "filed form number"),
        (re.compile(r"^form\s*\d{1,4}(-[A-Z])?$", re.I), "form reference"),
        (re.compile(r"^[A-Z]\d{7,10}$", re.I), "policy number"),
        (re.compile(r"^[A-TV-Z]\d{2}(\.\d{1,4})?$", re.I), "ICD-10 diagnosis code"),
    ],
    "fintech": [
        (re.compile(r"^\d{9}$"), "ABA routing number"),
        (re.compile(r"^mcc\s*\d{4}$", re.I), "merchant category code"),
        (re.compile(r"^\d{4}$"), "bare MCC"),
        (re.compile(r"^txn[-_]?[A-Z0-9]{6,}$", re.I), "transaction id"),
    ],
}

SAMPLE_QUERIES = {
    "retail": [
        "SKU-88213",
        "MFG#4471-B",
        "0714532198",
        "Salomon Quest 4 GTX",
        "Dyson V15 cordless",
        "patagonia fleece",
        "waterproof hiking boots size 11 wide",
        "insulated jacket 3 litre",
        "something warm for a rainy trek in Scotland",
        "best gift for a dad who camps",
        "watrproof hikking bots",
        "cordles vacum cleaner",
    ],
    "insurance": [
        "ICC21-TL-NY",
        "Form 10-R",
        "A0473321",
        "E11.9",
        "E11.29",
        "term life 20 year New York",
        "can an applicant with controlled hypertension qualify",
        "what is the contestability period for a reinstated policy",
        "accelerated death benefit rider exclusions",
    ],
    "fintech": [
        "MCC 5812",
        "021000021",
        "TXN-9F4B2C71",
        "AMZN MKTP US*2H4TR",
        "SQ *BLUE BOTTLE COFFEE",
        "why was my transaction declined",
        "dispute a duplicate charge on my statement",
        "chargeback timeline for card not present",
    ],
}


def print_route(r: Route) -> None:
    print(f"  query      : {r.query!r}")
    print(f"  class      : {r.query_class}")
    if r.short_circuit:
        print("  routing    : SHORT-CIRCUIT — no fusion")
    else:
        print(f"  weights    : lexical {r.w_lexical:.2f}  |  dense {r.w_dense:.2f}")
    print(f"  strategy   : {r.strategy}")
    for s in r.signals:
        print(f"    signal   - {s}")
    if r.note:
        # wrap the note at ~72 chars
        words, line = r.note.split(), ""
        lines = []
        for w in words:
            if len(line) + len(w) + 1 > 68:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        for i, ln in enumerate(lines):
            print(f"  {'note       : ' if i == 0 else '             '}{ln}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", type=str, default=None,
                   help="classify a single query instead of the sample set")
    p.add_argument("--domain", choices=sorted(SAMPLE_QUERIES), default="retail",
                   help="which domain's identifier grammar and sample set to use")
    args = p.parse_args()

    print("=" * 78)
    print(f"  QUERY CLASSIFIER — HYBRID ROUTING DECISIONS [{args.domain.upper()}]")
    print("=" * 78)
    print()

    queries = [args.query] if args.query else SAMPLE_QUERIES[args.domain]
    routes = [classify(q, args.domain) for q in queries]
    for r in routes:
        print_route(r)

    if not args.query:
        print("-" * 78)
        print("  CLASS DISTRIBUTION OVER THE SAMPLE SET")
        print("-" * 78)
        counts: dict[str, int] = {}
        for r in routes:
            counts[r.query_class] = counts.get(r.query_class, 0) + 1
        for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {cls:<24}{n:>3}")
        print()
        print("  Contrast with a single global alpha=0.5: every EXACT_IDENTIFIER")
        print("  query above would have had dense results blended into a perfect")
        print("  match, diluting rank 1 on your highest-converting traffic.")
        print("  Run hybrid_eval_by_class.py to see what that costs.")
        print("=" * 78)


if __name__ == "__main__":
    main()
