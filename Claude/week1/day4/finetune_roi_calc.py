"""
Week 1, Day 4 — Code 3: Fine-Tune vs Hybrid Search ROI Calculator

Pure Python — no libraries needed.
Input your team's actual numbers to get a go/no-go recommendation.
"""


def roi_analysis(
    # ── Training data availability ────────────────────────────────────────
    num_training_pairs: int     = 5_000,      # (query, positive) pairs you have

    # ── Expected nDCG@10 improvements (estimated from BEIR benchmarks) ────
    baseline_ndcg: float        = 0.60,       # off-the-shelf model on your domain
    hybrid_ndcg_delta: float    = 0.10,       # +10% from adding BM25 + RRF
    reranker_ndcg_delta: float  = 0.08,       # +8% from adding cross-encoder
    finetune_ndcg_delta: float  = 0.12,       # +12% from fine-tuning (optimistic)

    # ── Business impact mapping ──────────────────────────────────────────
    monthly_search_revenue: float = 500_000,  # monthly GMV from search channel
    ndcg_to_revenue_elasticity: float = 0.5,  # 1% nDCG = 0.5% revenue lift (retail avg)

    # ── Fine-tuning costs ─────────────────────────────────────────────────
    ml_engineer_day_rate: float  = 1_200,     # $1,200/day (fully loaded)
    finetune_setup_days: int     = 20,        # initial setup (data pipeline, training infra)
    finetune_monthly_maintain: float = 5_000, # ongoing MLOps per month (0.3 FTE)
    gpu_training_cost: float     = 800,       # one-time training run (A10G spot)

    # ── Hybrid search costs ───────────────────────────────────────────────
    hybrid_setup_days: int       = 3,
    hybrid_monthly_maintain: float = 500,     # Elasticsearch/OpenSearch operational cost
):
    print("=" * 70)
    print("   FINE-TUNE vs HYBRID SEARCH — ROI DECISION CALCULATOR")
    print("=" * 70)

    # ── Data gate check ─────────────────────────────────────────────────
    print(f"\n📊 DATA AVAILABILITY: {num_training_pairs:,} training pairs")
    if num_training_pairs < 5_000:
        print("  ⛔ INSUFFICIENT DATA — Fine-tuning will produce unreliable results.")
        print("     Minimum recommended: 10,000 high-quality (query, positive) pairs.")
        print("     RECOMMENDATION: Build hybrid search. Invest time in data collection.")
        return
    elif num_training_pairs < 10_000:
        print("  ⚠️  MARGINAL DATA — Fine-tuning possible but validation will be noisy.")
        finetune_ndcg_delta *= 0.6  # Discount expected gains
    else:
        print("  ✅ Sufficient training data for reliable fine-tuning.")

    # ── nDCG projections ────────────────────────────────────────────────
    hybrid_ndcg    = baseline_ndcg + hybrid_ndcg_delta
    reranked_ndcg  = hybrid_ndcg + reranker_ndcg_delta
    finetuned_ndcg = baseline_ndcg + finetune_ndcg_delta

    print(f"\n📈 nDCG@10 PROJECTIONS:")
    print(f"   Baseline (off-the-shelf):          {baseline_ndcg:.2f}")
    print(f"   + Hybrid BM25 + RRF:               {hybrid_ndcg:.2f}  (+{hybrid_ndcg_delta:.0%})")
    print(f"   + Hybrid + Cross-Encoder Reranker: {reranked_ndcg:.2f}  (+{reranker_ndcg_delta:.0%})")
    print(f"   Fine-Tuned Embedding (standalone):  {finetuned_ndcg:.2f}  (+{finetune_ndcg_delta:.0%})")

    # ── Revenue impact ───────────────────────────────────────────────────
    def revenue_lift(ndcg_delta):
        return monthly_search_revenue * ndcg_delta * ndcg_to_revenue_elasticity

    hybrid_rev   = revenue_lift(hybrid_ndcg_delta + reranker_ndcg_delta)
    finetune_rev = revenue_lift(finetune_ndcg_delta)

    print(f"\n💰 MONTHLY REVENUE IMPACT (elasticity = {ndcg_to_revenue_elasticity}× nDCG):")
    print(f"   Hybrid + Reranker revenue lift:  ${hybrid_rev:,.0f}/mo")
    print(f"   Fine-tune revenue lift:          ${finetune_rev:,.0f}/mo")

    # ── Cost comparison ──────────────────────────────────────────────────
    hybrid_setup_cost    = hybrid_setup_days * ml_engineer_day_rate
    finetune_setup_cost  = finetune_setup_days * ml_engineer_day_rate + gpu_training_cost

    print(f"\n🏗️  SETUP COSTS:")
    print(f"   Hybrid search setup:   ${hybrid_setup_cost:,.0f}  ({hybrid_setup_days} eng-days)")
    print(f"   Fine-tune setup:       ${finetune_setup_cost:,.0f}  ({finetune_setup_days} eng-days + GPU)")

    print(f"\n🔧 ONGOING MONTHLY COSTS:")
    print(f"   Hybrid search:         ${hybrid_monthly_maintain:,.0f}/mo")
    print(f"   Fine-tuned embedding:  ${finetune_monthly_maintain:,.0f}/mo")

    # ── Payback period ───────────────────────────────────────────────────
    hybrid_net_monthly  = max(1, hybrid_rev - hybrid_monthly_maintain)
    ft_net_monthly      = max(1, finetune_rev - finetune_monthly_maintain)

    hybrid_payback_months = hybrid_setup_cost / hybrid_net_monthly
    ft_payback_months     = finetune_setup_cost / ft_net_monthly

    print(f"\n⏱️  PAYBACK PERIOD:")
    print(f"   Hybrid + Reranker:     {hybrid_payback_months:.1f} months")
    print(f"   Fine-tuned Embedding:  {ft_payback_months:.1f} months")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    if hybrid_payback_months < ft_payback_months and reranked_ndcg >= finetuned_ndcg - 0.03:
        print("VERDICT: ✅ HYBRID SEARCH + RERANKER (Engineer B was right)")
        print("  Faster to ship, lower ongoing cost, comparable quality gain.")
    else:
        print("VERDICT: ⚠️ FINE-TUNING may be justified IF:")
        print("  - nDCG gap after hybrid+reranker is still >5%")
        print("  - You have dedicated ML infra team bandwidth")
        print("  - Training data quality is validated via offline eval")
    print("=" * 70)


if __name__ == "__main__":
    print("── Scenario A: Small team, 5K pairs ──────────────────────────────")
    roi_analysis(num_training_pairs=5_000)

    print("\n── Scenario B: Large team, 50K pairs, high-revenue search ───────")
    roi_analysis(
        num_training_pairs=50_000,
        monthly_search_revenue=5_000_000,
        finetune_ndcg_delta=0.18,
        finetune_setup_days=40,
        finetune_monthly_maintain=12_000,
    )
