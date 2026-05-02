"""
Evaluation utilities: aggregate metrics across debug results.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Sequence

from tot_debugger import DebugResult


# ── Per-result helpers ─────────────────────────────────────────────────────────

def result_to_dict(r: DebugResult) -> dict:
    return asdict(r) if hasattr(r, "__dataclass_fields__") else r.__dict__.copy()


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def compute_metrics(results: Sequence[DebugResult]) -> dict:
    """
    Returns a dict with:
      fix_rate              – fraction of problems solved
      first_attempt_rate   – fraction solved on first branch
      avg_tokens            – mean tokens per problem
      avg_nodes_explored    – mean search nodes per problem
      avg_backtracks        – mean backtracks (DFS)
      avg_time              – mean wall-clock seconds
    """
    n = len(results)
    if n == 0:
        return {}

    fix_rate = sum(r.success for r in results) / n
    first_rate = sum(r.first_attempt_success for r in results) / n
    avg_tokens = sum(r.total_tokens for r in results) / n
    avg_nodes = sum(r.nodes_explored for r in results) / n
    avg_bt = sum(r.backtracks for r in results) / n
    avg_time = sum(r.time_elapsed for r in results) / n

    return {
        "n": n,
        "fix_rate": round(fix_rate, 4),
        "first_attempt_rate": round(first_rate, 4),
        "avg_tokens": round(avg_tokens, 1),
        "avg_nodes_explored": round(avg_nodes, 2),
        "avg_backtracks": round(avg_bt, 3),
        "avg_time_sec": round(avg_time, 2),
    }


def compute_by_bug_type(results: Sequence[DebugResult]) -> dict[str, dict]:
    """Breakdown of fix_rate per bug category."""
    groups: dict[str, list[DebugResult]] = defaultdict(list)
    for r in results:
        groups[r.bug_type].append(r)
    return {bt: compute_metrics(rs) for bt, rs in sorted(groups.items())}


def compare_methods(all_results: dict[str, list[DebugResult]]) -> dict:
    """
    Given {method_name: [DebugResult]}, return comparison table.
    """
    return {method: compute_metrics(rs) for method, rs in all_results.items()}


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_results(results: list[DebugResult], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump([result_to_dict(r) for r in results], f, indent=2)
    print(f"Saved {len(results)} results → {path}")


def load_results(path: str) -> list[DebugResult]:
    with open(path) as f:
        data = json.load(f)
    return [DebugResult(**d) for d in data]


def print_comparison_table(comparison: dict):
    """Pretty-print a method-comparison table to stdout."""
    methods = list(comparison.keys())
    if not methods:
        print("No results to display.")
        return

    metrics = [
        ("fix_rate", "Fix Rate"),
        ("first_attempt_rate", "1st-Attempt Rate"),
        ("avg_tokens", "Avg Tokens"),
        ("avg_nodes_explored", "Avg Nodes"),
        ("avg_backtracks", "Avg Backtracks"),
        ("avg_time_sec", "Avg Time (s)"),
    ]

    col_w = max(len(m) for m in methods) + 2
    header_w = max(len(label) for _, label in metrics) + 2

    header = f"{'Metric':<{header_w}}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for key, label in metrics:
        row = f"{label:<{header_w}}"
        for m in methods:
            val = comparison[m].get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:>{col_w}.3f}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)

    print("=" * len(header))
