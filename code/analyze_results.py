"""
Load experiment results and produce comparison figures/tables.

Usage
-----
    python analyze_results.py --results_dir ../results/
    python analyze_results.py --results_dir ../results/ --save_figs
"""
from __future__ import annotations

import argparse
import json
import os
import glob

from evaluate import load_results, compare_methods, compute_by_bug_type, print_comparison_table


def load_all_results(results_dir: str) -> dict:
    all_results = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "summary":
            continue
        try:
            all_results[name] = load_results(path)
        except Exception as e:
            print(f"  Could not load {path}: {e}")
    return all_results


def plot_fix_rates(comparison: dict, save_path: str | None = None):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping plots. Run: pip install matplotlib")
        return

    methods = list(comparison.keys())
    fix_rates = [comparison[m]["fix_rate"] * 100 for m in methods]
    first_rates = [comparison[m].get("first_attempt_rate", 0) * 100 for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, fix_rates, width, label="Fix Rate (%)", color="#2196F3")
    bars2 = ax.bar(x + width / 2, first_rates, width, label="1st-Attempt Rate (%)", color="#FF9800")

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Fix Success Rate by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved fix-rate plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_token_cost(comparison: dict, save_path: str | None = None):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    methods = list(comparison.keys())
    tokens = [comparison[m]["avg_tokens"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(methods, tokens, color="#9C27B0", alpha=0.8)
    ax.set_ylabel("Avg Tokens per Problem")
    ax.set_title("Token Cost by Method")
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved token-cost plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_bug_type_breakdown(all_results: dict, method: str, save_path: str | None = None):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if method not in all_results:
        print(f"  Method '{method}' not found in results.")
        return

    by_type = compute_by_bug_type(all_results[method])
    bug_types = list(by_type.keys())
    rates = [by_type[bt]["fix_rate"] * 100 for bt in bug_types]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    ax.barh(bug_types, rates, color=colors[: len(bug_types)], alpha=0.85)
    ax.set_xlabel("Fix Rate (%)")
    ax.set_title(f"Fix Rate by Bug Type ({method})")
    ax.set_xlim(0, 105)
    for i, v in enumerate(rates):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved bug-type breakdown → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="../results/", help="Directory with *.json result files")
    parser.add_argument("--save_figs", action="store_true", help="Save figures to results_dir")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir} ...")
    all_results = load_all_results(args.results_dir)

    if not all_results:
        print("No result files found. Run run_experiments.py first.")
        return

    comparison = compare_methods(all_results)
    print_comparison_table(comparison)

    figs_dir = args.results_dir if args.save_figs else None

    fix_rate_path = os.path.join(figs_dir, "fix_rates.png") if figs_dir else None
    token_path = os.path.join(figs_dir, "token_cost.png") if figs_dir else None

    plot_fix_rates(comparison, save_path=fix_rate_path)
    plot_token_cost(comparison, save_path=token_path)

    # Bug-type breakdown for first ToT method found
    tot_methods = [m for m in all_results if m.startswith("tot")]
    if tot_methods:
        bt_path = os.path.join(figs_dir, "bug_type_breakdown.png") if figs_dir else None
        plot_bug_type_breakdown(all_results, tot_methods[0], save_path=bt_path)


if __name__ == "__main__":
    main()
