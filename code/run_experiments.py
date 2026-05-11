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
from data_loader import load_humaneval_bugs, load_debugbench, load_curated_problems, load_game24, Problem
from tot_debugger import ToTDebugger, DebugResult
from baselines import IOBaseline, CoTBaseline, CoTSCBaseline
from evaluate import (
    compute_metrics,
    compute_by_bug_type,
    compare_methods,
    save_results,
    print_comparison_table,
)
from game24_solver import (
    Game24Solver,
    Game24IOBaseline,
    Game24CoTBaseline,
    Game24CoTSCBaseline,
    run_game24,
    compute_game24_metrics,
    print_game24_table,
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


def _run_game24_task(args, verbose: bool):
    """
    Replicate Yao et al. (NeurIPS 2023) Table 2 — Game of 24.
    Paper result: IO=7.3%, CoT=4%, CoT-SC(k=100)=9%, ToT-BFS(b=5)=74%.
    """
    print("=" * 60)
    print("Game of 24  —  Tree of Thoughts paper replication")
    print("=" * 60)

    # ── LLM ──────────────────────────────────────────────────────────────────
    if args.demo:
        from llm_client import MockLLMClient
        llm = MockLLMClient()
        print("[game24] Using MockLLMClient (offline demo).")
    elif config.BACKEND == "ollama":
        llm = LLMClient()
        print(f"[game24] Using Ollama → {config.MODEL}")
    else:
        llm = LLMClient()
        print(f"[game24] Using {config.BACKEND} → {config.MODEL}")

    # ── Load puzzles ──────────────────────────────────────────────────────────
    n = args.n or 100
    puzzles = load_game24(n=n, csv_path=args.game24_csv, seed=args.seed)
    print(f"[game24] {len(puzzles)} puzzles loaded\n")

    os.makedirs(args.out, exist_ok=True)
    all_results: dict[str, list] = {}

    # ── ToT ───────────────────────────────────────────────────────────────────
    searches = []
    if args.both or args.search == "both":
        searches = ["bfs", "dfs"]
    else:
        searches = [args.search] if args.search in ("bfs", "dfs") else ["bfs"]

    for search in searches:
        label = f"tot-{search}-b{args.k}"
        print(f"── Running {label} ──")
        solver = Game24Solver(llm, b=args.k, search=search)
        results = run_game24(puzzles, solver, label, verbose)
        all_results[label] = results
        m = compute_game24_metrics(results)
        print(f"  success={m['success_rate']:.1%}  avg_tokens={m['avg_tokens']:.0f}\n")

        with open(os.path.join(args.out, f"game24_{label}.json"), "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)

    # ── Baselines ─────────────────────────────────────────────────────────────
    if args.baselines or args.demo:
        for label, solver in [
            ("io", Game24IOBaseline(llm)),
            ("cot", Game24CoTBaseline(llm)),
            (f"cot-sc-5", Game24CoTSCBaseline(llm, n_samples=5)),
        ]:
            print(f"── Running {label} ──")
            results = run_game24(puzzles, solver, label, verbose)
            all_results[label] = results
            m = compute_game24_metrics(results)
            print(f"  success={m['success_rate']:.1%}  avg_tokens={m['avg_tokens']:.0f}\n")

            with open(os.path.join(args.out, f"game24_{label}.json"), "w") as f:
                json.dump([r.__dict__ for r in results], f, indent=2)

    print_game24_table(all_results)


def main():
    parser = argparse.ArgumentParser(description="ToT Experiments (code debugging + Game of 24)")
    parser.add_argument("--demo", action="store_true", help="Run offline demo with mock LLM")
    parser.add_argument("--task", choices=["debug", "game24"], default="debug",
                        help="Task: 'debug' (code debugging) or 'game24' (Game of 24 paper replication)")
    parser.add_argument("--n", type=int, default=config.NUM_PROBLEMS, help="Number of problems")
    parser.add_argument("--k", type=int, default=config.TOT_K, help="Branching factor / beam width")
    parser.add_argument("--search", choices=["bfs", "dfs", "mcts", "both"], default="bfs")
    parser.add_argument("--evaluator", choices=["llm", "execution", "hybrid"], default=config.EVALUATOR)
    parser.add_argument("--baselines", action="store_true", help="Also run IO, CoT, CoT-SC baselines")
    parser.add_argument("--both", action="store_true", help="Run both BFS and DFS")
    parser.add_argument("--dataset", choices=["humaneval", "debugbench", "curated"], default="humaneval",
                        help="Debug task dataset (default: humaneval)")
    parser.add_argument("--game24-csv", type=str, default=None,
                        help="Path to 24.csv from princeton-nlp/tree-of-thought-llm (for exact replication)")
    parser.add_argument("--bug-types", nargs="+", default=["Logic Error"],
                        help="DebugBench bug type filter (default: Logic Error)")
    parser.add_argument("--out", type=str, default=config.RESULTS_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    # ── Game of 24 task (paper replication) ───────────────────────────────────
    if args.task == "game24":
        _run_game24_task(args, verbose)
        return

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
    print(f"[run_experiments] Loading {args.n} problems from {args.dataset} (seed={args.seed})...")
    if args.dataset == "debugbench":
        problems = load_debugbench(n=args.n, bug_types=args.bug_types, seed=args.seed)
    elif args.dataset == "curated":
        problems = load_curated_problems()
        if args.n:
            problems = problems[:args.n]
    else:
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
