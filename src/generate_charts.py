import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure figures directory exists in the project root
os.makedirs("figures", exist_ok=True)

def generate_break_even_chart():
    # 1. Break-Even Analysis Chart
    # Assume we analyze monthly usage up to 1 Billion tokens
    tokens_millions = np.linspace(0, 1000, 100) 
    
    # API Cost: Assuming a 14B model is equivalent to a smaller, cheaper cloud model like GPT-3.5 / Haiku
    # Let's say $0.50 per 1M tokens (blended input/output)
    api_cost_per_million = 0.50 
    api_costs = tokens_millions * api_cost_per_million

    # On-Premises Cost:
    # CAPEX: Consumer machine / small server ~ $2000, amortized over 36 months = ~$55/mo
    # OPEX: Electricity & minimal maintenance = ~$15/mo
    # Total Fixed Monthly = $70/mo
    on_prem_fixed = 70
    on_prem_costs = [on_prem_fixed] * len(tokens_millions)

    plt.figure(figsize=(10, 6))
    plt.plot(tokens_millions, api_costs, label="API Cost (Cloud/Third-party)", color="blue", linewidth=2)
    plt.plot(tokens_millions, on_prem_costs, label="On-Premises Cost (Local)", color="red", linewidth=2, linestyle="--")

    # Find intersection
    break_even = on_prem_fixed / api_cost_per_million
    plt.axvline(x=break_even, color="gray", linestyle=":", label=f"Break-Even Point\n({break_even:.0f}M tokens/mo)")
    plt.plot(break_even, on_prem_fixed, 'ko') # intersection point

    plt.title("Economic Viability Analysis: On-Premises vs API", fontsize=14, pad=15)
    plt.xlabel("Usage (Millions of Tokens per Month)", fontsize=12)
    plt.ylabel("Monthly Cost ($ USD)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add context caching annotation
    plt.annotate("Prompt/Context Caching in APIs\ncould push the API cost curve lower,\nshifting the break-even point right.",
                 xy=(500, 250), xytext=(550, 150),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig("figures/break_even.png", dpi=150)
    print("Saved figures/break_even.png")

def generate_benchmark_chart():
    # 2. Benchmarks Bar Chart (TTFT and Peak RAM)
    frameworks = ["Standard HF", "Ollama (Q4)", "AirLLM"]
    ram = [28.5, 0.25, 2.03]
    ttft = [0, 18.23, 1014.75] # Standard HF crashed, represented as 0
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(frameworks))
    width = 0.35

    # Plot RAM
    ax1.set_xlabel('Framework', fontsize=12)
    ax1.set_ylabel('Peak RAM (GB)', color='tab:red', fontsize=12)
    bars1 = ax1.bar(x - width/2, ram, width, label='Peak RAM (GB)', color='tab:red', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(0, 32)
    
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f} GB", ha='center', va='bottom', fontsize=9, color='darkred')

    # Plot TTFT
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Time to First Token (s)', color='tab:blue', fontsize=12)
    bars2 = ax2.bar(x + width/2, ttft, width, label='TTFT (s)', color='tab:blue', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 1200)

    # Annotate AirLLM TTFT
    ax2.text(x[2] + width/2, ttft[2] + 20, f"{ttft[2]:.1f} s", ha='center', va='bottom', fontsize=9, color='darkblue')

    # Annotate Standard HF crash
    ax2.text(0 + width/2, 50, "OOM CRASH", ha='center', va='bottom', rotation=90, color='black', fontweight='bold')

    plt.title("LLM Inference Benchmark: Peak RAM vs TTFT", fontsize=14, pad=15)
    plt.xticks(x, frameworks, fontsize=11)
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()
    plt.savefig("figures/benchmarks.png", dpi=150)
    print("Saved figures/benchmarks.png")

if __name__ == "__main__":
    generate_break_even_chart()
    generate_benchmark_chart()
