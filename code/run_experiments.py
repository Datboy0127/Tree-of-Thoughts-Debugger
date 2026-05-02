"""
Main experiment runner.

Usage
-----
# Demo (no API key needed, uses mock LLM + built-in problems):
    python run_experiments.py --demo

# Full run (requires OPENAI_API_KEY env var):
    python run_experiments.py --n 50 --k 3 --search bfs

# Both BFS and DFS, save results:
    python run_experiments.py --n 50 --k 3 --both --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import config
from llm_client import LLMClient, MockLLMClient
from executor import CodeExecutor
from data_loader import load_humaneval_bugs, Problem
from tot_debugger import ToTDebugger, DebugResult
from baselines import IOBaseline, CoTBaseline, CoTSCBaseline
from evaluate import (
    compute_metrics,
    compute_by_bug_type,
    compare_methods,
    save_results,
    print_comparison_table,
)


def run_method(solver, problems: list[Problem], label: str, verbose: bool = True) -> list[DebugResult]:
    results = []
    for i, problem in enumerate(problems):
        try:
            result = solver.solve(problem)
        except Exception as e:
            print(f"  [{label}] ERROR on {problem.task_id}: {e}")
            result = DebugResult(
                task_id=problem.task_id,
                method=label,
                success=False,
                fix_code=None,
                nodes_explored=0,
                backtracks=0,
                total_tokens=0,
                time_elapsed=0.0,
                first_attempt_success=False,
                bug_type=problem.bug_type,
            )
        results.append(result)
        if verbose:
            status = "✓" if result.success else "✗"
            print(f"  [{label}] {status} {problem.task_id:20s}  tokens={result.total_tokens:5d}  nodes={result.nodes_explored}")
    return results


def main():
    parser = argparse.ArgumentParser(description="ToT Code Debugger Experiments")
    parser.add_argument("--demo", action="store_true", help="Run offline demo with mock LLM")
    parser.add_argument("--n", type=int, default=config.NUM_PROBLEMS, help="Number of problems")
    parser.add_argument("--k", type=int, default=config.TOT_K, help="Branching factor")
    parser.add_argument("--search", choices=["bfs", "dfs", "both"], default="bfs")
    parser.add_argument("--evaluator", choices=["llm", "execution", "hybrid"], default=config.EVALUATOR)
    parser.add_argument("--baselines", action="store_true", help="Also run IO, CoT, CoT-SC baselines")
    parser.add_argument("--both", action="store_true", help="Run both BFS and DFS")
    parser.add_argument("--out", type=str, default=config.RESULTS_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    # ── Setup LLM ─────────────────────────────────────────────────────────────
    if args.demo:
        print("[run_experiments] Using MockLLMClient (offline demo mode).")
        llm = MockLLMClient()
    elif config.BACKEND == "ollama":
        print(f"[run_experiments] Using Ollama backend → {config.MODEL}")
        print("  Make sure Ollama is running: ollama serve")
        llm = LLMClient()
    elif not config.OPENAI_API_KEY:
        print("[run_experiments] No API key found. Falling back to MockLLMClient.")
        print("  Set OPENAI_API_KEY or use --demo for offline mode.")
        llm = MockLLMClient()
    else:
        print(f"[run_experiments] Using {config.BACKEND} backend → {config.MODEL}")
        llm = LLMClient()

    executor = CodeExecutor()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"[run_experiments] Loading {args.n} problems (seed={args.seed})...")
    problems = load_humaneval_bugs(n=args.n, use_synthetic=True, seed=args.seed)
    print(f"  Loaded {len(problems)} problems: {set(p.bug_type for p in problems)}\n")

    os.makedirs(args.out, exist_ok=True)
    all_results: dict[str, list[DebugResult]] = {}

    # ── Run ToT ────────────────────────────────────────────────────────────────
    searches = []
    if args.both:
        searches = ["bfs", "dfs"]
    elif args.search == "both":
        searches = ["bfs", "dfs"]
    else:
        searches = [args.search]

    for search in searches:
        label = f"tot-{search}-k{args.k}"
        print(f"── Running {label} ──")
        solver = ToTDebugger(llm, executor, k=args.k, search=search, evaluator=args.evaluator)
        results = run_method(solver, problems, label, verbose)
        all_results[label] = results
        save_results(results, os.path.join(args.out, f"{label}.json"))
        metrics = compute_metrics(results)
        print(f"  fix_rate={metrics['fix_rate']:.1%}  avg_tokens={metrics['avg_tokens']:.0f}\n")

    # ── Run baselines ──────────────────────────────────────────────────────────
    if args.baselines or args.demo:
        for label, solver in [
            ("io", IOBaseline(llm, executor)),
            ("cot", CoTBaseline(llm, executor)),
            (f"cot-sc-{config.COT_SC_SAMPLES}", CoTSCBaseline(llm, executor, config.COT_SC_SAMPLES)),
        ]:
            print(f"── Running {label} ──")
            results = run_method(solver, problems, label, verbose)
            all_results[label] = results
            save_results(results, os.path.join(args.out, f"{label}.json"))
            metrics = compute_metrics(results)
            print(f"  fix_rate={metrics['fix_rate']:.1%}  avg_tokens={metrics['avg_tokens']:.0f}\n")

    # ── Summary table ──────────────────────────────────────────────────────────
    comparison = compare_methods(all_results)
    print_comparison_table(comparison)

    # Save comparison summary
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    # Bug-type breakdown for first method
    if all_results:
        first_method = next(iter(all_results))
        by_type = compute_by_bug_type(all_results[first_method])
        print(f"\n── Bug-type breakdown ({first_method}) ──")
        for bt, m in by_type.items():
            print(f"  {bt:25s}  fix_rate={m['fix_rate']:.1%}  n={m['n']}")


if __name__ == "__main__":
    main()
