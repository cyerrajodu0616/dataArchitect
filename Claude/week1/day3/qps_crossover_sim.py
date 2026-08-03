"""
Week 1, Day 3 — Deep Dive Code 3: Production QPS Crossover & Cost Simulator

Simulates monthly costs of OpenAI Embedding API vs. Self-Hosted GPU TEI Container
based on input QPS, token lengths, GPU instance hourly rates, and dataset sizes.

Run:
  python Claude/week1/day3/qps_crossover_sim.py
"""

def calculate_costs(
    num_skus: int = 10_000_000,
    sku_token_len: int = 100,
    query_token_len: int = 15,
    qps_target: float = 500.0,
    openai_batch_rate_per_m: float = 0.01,
    openai_realtime_rate_per_m: float = 0.02,
    gpu_hourly_cost: float = 1.006,  # AWS g5.xlarge (1x A10G GPU)
    gpu_max_qps_capacity: int = 1200
):
    print("="*65)
    print("      ENTERPRISE EMBEDDING COST & QPS CROSSOVER CALCULATOR      ")
    print("="*65)
    print(f"Catalog Size       : {num_skus:,} SKUs ({sku_token_len} tokens/SKU)")
    print(f"Real-Time Traffic  : {qps_target} Queries/Sec ({query_token_len} tokens/query)")
    print(f"GPU Host Hardware  : AWS g5.xlarge (${gpu_hourly_cost:.3f}/hr)")
    print("-" * 65)

    # 1. Batch Monthly Re-embed Cost
    total_batch_tokens = num_skus * sku_token_len
    batch_cost_openai = (total_batch_tokens / 1_000_000) * openai_batch_rate_per_m

    # 2. Real-time Monthly Query Cost
    seconds_in_month = 30 * 24 * 3600  # 2,592,000 seconds
    monthly_queries = qps_target * seconds_in_month
    total_query_tokens = monthly_queries * query_token_len
    realtime_cost_openai = (total_query_tokens / 1_000_000) * openai_realtime_rate_per_m

    total_openai_monthly = batch_cost_openai + realtime_cost_openai

    # 3. Self-Hosted GPU Cost
    # Calculate GPUs required
    gpus_needed = max(1, int((qps_target + gpu_max_qps_capacity - 1) // gpu_max_qps_capacity))
    total_gpu_monthly = gpus_needed * (gpu_hourly_cost * 24 * 30)

    # 4. Exact Crossover QPS Derivation
    # Cost_API(Q) = Cost_GPU_Base (724.32)
    # Q * 15 * 2,592,000 / 1,000,000 * 0.02 = 724.32
    tokens_per_qps_month = 15 * seconds_in_month
    api_cost_per_qps = (tokens_per_qps_month / 1_000_000) * openai_realtime_rate_per_m
    crossover_qps = (gpu_hourly_cost * 24 * 30) / api_cost_per_qps

    print(f"1. OpenAI Monthly Cost Breakdown:")
    print(f"   - Catalog Re-embed Batch (1B tokens) : ${batch_cost_openai:,.2f}")
    print(f"   - Real-Time Queries ({total_query_tokens/1e6:.1f}M tokens): ${realtime_cost_openai:,.2f}")
    print(f"   -> TOTAL OPENAI COST                 : ${total_openai_monthly:,.2f} / month")
    print(f"\n2. Self-Hosted GPU (TEI) Monthly Cost:")
    print(f"   - GPUs Required ({gpus_needed}x A10G)           : ${total_gpu_monthly:,.2f} / month")
    print("-" * 65)

    if total_openai_monthly < total_gpu_monthly:
        savings = total_gpu_monthly - total_openai_monthly
        print(f"VERDICT at {qps_target} QPS : OPENAI API WINS (Saves ${savings:,.2f}/mo over self-hosting)")
    else:
        savings = total_openai_monthly - total_gpu_monthly
        print(f"VERDICT at {qps_target} QPS : SELF-HOSTED GPU WINS (Saves ${savings:,.2f}/mo over OpenAI)")

    print(f"\nEXACT CROSSOVER THRESHOLD: {crossover_qps:.1f} QPS")
    print("  • Below 931 QPS -> Managed OpenAI API is cheaper.")
    print("  • Above 931 QPS -> Self-hosted GPU container is cheaper.")
    print("="*65)

if __name__ == "__main__":
    calculate_costs(qps_target=500.0)
