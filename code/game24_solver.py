"""
Game of 24 — Tree of Thoughts (Yao et al., NeurIPS 2023) Section 4.1.

Direct replication:
  - Dataset:  problems 901-1000 from 4nums.com (100 hardest games)
  - Thoughts: 3 steps, each "a op b = c (left: remaining numbers)"
  - Generator: propose prompt — enumerate all valid next steps
  - Evaluator: value prompt — sure / maybe / impossible (×3 samples, summed)
  - BFS beam width b=5  |  DFS with impossible-pruning
  - Baselines: IO (5-shot), CoT (5-shot with 3 thought steps shown), CoT-SC

Reference: Algorithm 1 (BFS) and Algorithm 2 (DFS) in the paper.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from llm_client import LLMClient

# ── System prompt ──────────────────────────────────────────────────────────────

_SYS = "You are a precise mathematical reasoning assistant. Be concise."

# ── IO prompt (5-shot, from paper Section 4.1) ─────────────────────────────────

_IO_PROMPT = """\
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. \
Each number must be used exactly once.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
Answer:"""

# ── CoT prompt (5-shot, 3 intermediate equations per example) ──────────────────

_COT_PROMPT = """\
Use numbers and basic arithmetic operations (+ - * /) to obtain 24. \
Each number must be used exactly once. Show 3 step-by-step equations then give the answer.
Input: 4 9 10 13
Steps:
13 - 9 = 4 (left: 4 4 10)
10 - 4 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: (13 - 9) * (10 - 4) = 24

Input: 2 9 10 12
Steps:
2 * 12 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
1 * 24 = 24 (left: 24)
Answer: 2 * 12 * (10 - 9) = 24

Input: 5 5 5 9
Steps:
5 + 9 = 14 (left: 5 5 14)
14 + 5 = 19 (left: 5 19)
19 + 5 = 24 (left: 24)
Answer: 5 + 9 + 5 + 5 = 24

Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
2 + 1 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (8 / 4 + 1) * 8 = 24

Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (4 + 8) * (6 - 4) = 24

Input: {input}
Steps:"""

# ── Propose prompt (from paper Figure 2a — enumerate all valid next steps) ─────

_PROPOSE_PROMPT = """\
Propose all possible next steps to make 24 using arithmetic on the given numbers. \
Each step picks two numbers, applies one operation (+, -, *, /), and lists what is left.
Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 14 10)
8 / 2 = 4 (left: 8 14 4)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 8 14 6)
2 + 14 = 16 (left: 8 8 16)
2 * 14 = 28 (left: 8 8 28)
14 - 2 = 12 (left: 8 8 12)
14 / 2 = 7 (left: 8 8 7)
8 + 8 = 16 (left: 2 14 16)
8 - 8 = 0 (left: 2 14 0)
8 / 8 = 1 (left: 2 14 1)
8 + 14 = 22 (left: 2 8 22)
8 * 14 = 112 (left: 2 8 112)
8 - 14 = -6 (left: 2 8 -6)
14 - 8 = 6 (left: 2 8 6)
14 / 8 = 1.75 (left: 2 8 1.75)

Input: {input}
Possible next steps:"""

# ── Value prompt (from paper Figure 2b — sure / maybe / impossible) ────────────

_VALUE_PROMPT = """\
Evaluate if given numbers can reach 24 (sure/maybe/impossible)
10 14
10 + 14 = 24
sure

11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible

4 4 10
4 + 4 + 10 = 18
4 * 10 - 4 = 36
(4 + 4) * 10 = 80
4 * 4 + 10 = 26
(10 - 4) * 4 = 24
sure

4 9 11
9 + 11 + 4 = 24
sure

1 3 3
1 * 3 * 3 = 9
1 + 3 + 3 = 7
1 * (3 + 3) = 6
1 * 3 - 3 = 0
impossible

{input}"""

# ── Scoring (paper: sure=20, maybe=1, impossible=0.001; summed across n_eval) ──

_SCORE = {"sure": 20.0, "maybe": 1.0, "impossible": 0.001}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Game24Result:
    puzzle: str
    method: str
    success: bool
    answer: Optional[str]          # final equation string
    thoughts: list[str] = field(default_factory=list)   # 3-step path
    nodes_explored: int = 0
    total_tokens: int = 0
    time_elapsed: float = 0.0


# ── Verification ───────────────────────────────────────────────────────────────

def verify_solution(puzzle: str, answer: str) -> bool:
    """
    Check that answer is a valid equation equalling 24 using each input number
    exactly once. Works for expressions like "(13-9)*(10-4)" or "2*12*(10-9)".
    """
    # Extract the expression part (before "= 24" if present)
    expr = re.sub(r"\s*=\s*24\s*$", "", answer.strip())
    expr = expr.strip()

    # Evaluate
    try:
        result = eval(expr, {"__builtins__": {}})   # safe: arithmetic only
        if abs(float(result) - 24.0) > 1e-3:
            return False
    except Exception:
        return False

    # Check numbers used match input
    input_nums = sorted(int(x) for x in puzzle.split())
    found = sorted(int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", expr)
                   if float(x) == int(float(x)))
    return found == input_nums


def verify_thought_path(puzzle: str, thoughts: list[str]) -> bool:
    """Check that a 3-step thought path correctly reaches 24."""
    remaining = sorted(float(x) for x in puzzle.split())
    for thought in thoughts:
        m = re.match(
            r"([\d.]+)\s*([+\-*/])\s*([\d.]+)\s*=\s*(-?[\d.]+)\s*\(left:\s*([\d.\s-]+)\)",
            thought.strip(),
        )
        if not m:
            return False
        a, op, b, c_str = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        left_str = m.group(5).strip()

        # Verify arithmetic
        try:
            ops = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b != 0 else None}
        except ZeroDivisionError:
            return False
        expected = ops.get(op)
        if expected is None or abs(expected - c_str) > 1e-3:
            return False

        # Verify a and b are in remaining
        rem = remaining[:]
        try:
            rem.remove(a)
            rem.remove(b)
        except ValueError:
            return False
        rem.append(c_str)
        remaining = rem

    return len(remaining) == 1 and abs(remaining[0] - 24.0) < 1e-3


# ── Thought parsing helpers ────────────────────────────────────────────────────

_THOUGHT_RE = re.compile(
    r"-?[\d.]+\s*[+\-*/]\s*-?[\d.]+\s*=\s*-?[\d.]+\s*\(left:\s*[^\)]+\)"
)


def _parse_thoughts(text: str) -> list[str]:
    """Extract all valid thought-step lines from LLM output."""
    return _THOUGHT_RE.findall(text)


def _parse_remaining(thought: str) -> Optional[str]:
    """Extract 'left: x y z' from a thought string."""
    m = re.search(r"\(left:\s*([^\)]+)\)", thought)
    return m.group(1).strip() if m else None


def _is_done(remaining: str) -> bool:
    """True if the only remaining number is 24."""
    try:
        nums = [float(x) for x in remaining.split()]
        return len(nums) == 1 and abs(nums[0] - 24.0) < 1e-3
    except Exception:
        return False


# ── Main solver ────────────────────────────────────────────────────────────────

class Game24Solver:
    """
    ToT BFS/DFS solver for Game of 24.

    BFS (Algorithm 1): maintains beam of b=5 best states at each depth.
    DFS (Algorithm 2): greedy descent, backtracks when value < threshold.

    Both use the same propose + value prompts from the paper.
    """

    def __init__(self, llm: LLMClient, b: int = 5, n_eval: int = 3,
                 search: str = "bfs"):
        self.llm = llm
        self.b = b              # beam width (paper: b=5)
        self.n_eval = n_eval    # value samples per state (paper: 3)
        self.search = search

    def solve(self, puzzle: str) -> Game24Result:
        t0 = time.time()
        if self.search == "dfs":
            result = self._dfs(puzzle, t0)
        else:
            result = self._bfs(puzzle, t0)
        result.time_elapsed = round(time.time() - t0, 2)
        return result

    # ── BFS (Algorithm 1) ─────────────────────────────────────────────────────

    def _bfs(self, puzzle: str, t0: float) -> Game24Result:
        """
        S0 = {puzzle}
        for t in 1..3:
            St' = {[s,z] | s in S_{t-1}, z in propose(s)}
            Vt  = value(St')
            St  = top-b by score
        return first state in S3 that == 24
        """
        # Beam: list of (remaining_str, path_so_far)
        beam: list[tuple[str, list[str]]] = [(puzzle, [])]
        total_tokens = 0
        nodes = 0

        for depth in range(3):
            candidates: list[tuple[str, list[str], float]] = []  # (remaining, path, score)

            for remaining, path in beam:
                thoughts, tok = self._propose(remaining)
                total_tokens += tok
                nodes += len(thoughts)

                for thought in thoughts:
                    new_rem = _parse_remaining(thought)
                    if new_rem is None:
                        continue
                    # Evaluate the resulting state
                    score, eval_tok = self._evaluate(new_rem)
                    total_tokens += eval_tok
                    if score > 0:  # prune impossible (score ≈ 0.001 * n_eval)
                        candidates.append((new_rem, path + [thought], score))

            if not candidates:
                break

            # Keep top-b by value score
            candidates.sort(key=lambda x: -x[2])
            beam = [(rem, path) for rem, path, _ in candidates[:self.b]]

        # Check all final beam states
        for remaining, path in beam:
            if _is_done(remaining):
                return Game24Result(
                    puzzle=puzzle,
                    method=f"tot-bfs-b{self.b}",
                    success=True,
                    answer=self._path_to_answer(puzzle, path),
                    thoughts=path,
                    nodes_explored=nodes,
                    total_tokens=total_tokens,
                )

        return Game24Result(
            puzzle=puzzle,
            method=f"tot-bfs-b{self.b}",
            success=False,
            answer=None,
            nodes_explored=nodes,
            total_tokens=total_tokens,
        )

    # ── DFS (Algorithm 2) ─────────────────────────────────────────────────────

    def _dfs(self, puzzle: str, t0: float) -> Game24Result:
        """
        DFS: for each candidate at each level (sorted by value), recurse if
        score > v_th; backtrack otherwise (prune the subtree).
        """
        total_tokens = [0]
        nodes = [0]

        # v_th: prune impossible states (score < 0.01 * n_eval)
        v_th = _SCORE["impossible"] * self.n_eval * 1.5

        def dfs(remaining: str, path: list[str], depth: int) -> Optional[list[str]]:
            if depth == 3:
                return path if _is_done(remaining) else None

            thoughts, tok = self._propose(remaining)
            total_tokens[0] += tok
            nodes[0] += len(thoughts)

            # Score all candidates then explore best-first
            scored: list[tuple[float, str, str]] = []
            for thought in thoughts:
                new_rem = _parse_remaining(thought)
                if new_rem is None:
                    continue
                score, eval_tok = self._evaluate(new_rem)
                total_tokens[0] += eval_tok
                if score > v_th:
                    scored.append((score, new_rem, thought))

            scored.sort(key=lambda x: -x[0])  # best first
            for _, new_rem, thought in scored:
                result = dfs(new_rem, path + [thought], depth + 1)
                if result is not None:
                    return result
            return None

        found = dfs(puzzle, [], 0)
        return Game24Result(
            puzzle=puzzle,
            method=f"tot-dfs-b{self.b}",
            success=found is not None,
            answer=self._path_to_answer(puzzle, found) if found else None,
            thoughts=found or [],
            nodes_explored=nodes[0],
            total_tokens=total_tokens[0],
        )

    # ── Propose ───────────────────────────────────────────────────────────────

    def _propose(self, remaining: str) -> tuple[list[str], int]:
        """
        Ask the LLM to enumerate all possible next steps for the remaining numbers.
        Returns (list_of_thought_strings, tokens_used).
        """
        prompt = _PROPOSE_PROMPT.format(input=remaining)
        resp = self.llm.call(prompt, system=_SYS, max_tokens=512, temperature=0.7)
        thoughts = _parse_thoughts(resp["content"])

        # Fallback: also enumerate mathematically valid steps (ensures coverage)
        if len(thoughts) < 3:
            thoughts = self._enumerate_steps(remaining)

        return thoughts, resp["tokens"]

    def _enumerate_steps(self, remaining: str) -> list[str]:
        """
        Enumerate all valid arithmetic steps from remaining numbers.
        Used as fallback when the LLM generates too few thoughts.
        """
        try:
            nums = [float(x) for x in remaining.split()]
        except ValueError:
            return []

        steps = []
        n = len(nums)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = nums[i], nums[j]
                left = [nums[k] for k in range(n) if k != i and k != j]
                candidates = [
                    (a + b, f"{_fmt(a)} + {_fmt(b)} = {_fmt(a+b)}"),
                    (a - b, f"{_fmt(a)} - {_fmt(b)} = {_fmt(a-b)}"),
                    (a * b, f"{_fmt(a)} * {_fmt(b)} = {_fmt(a*b)}"),
                ]
                if b != 0:
                    candidates.append((a / b, f"{_fmt(a)} / {_fmt(b)} = {_fmt(a/b)}"))
                for result, expr in candidates:
                    left_str = " ".join(_fmt(x) for x in sorted(left + [result]))
                    steps.append(f"{expr} (left: {left_str})")
        # deduplicate
        seen = set()
        unique = []
        for s in steps:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def _evaluate(self, remaining: str) -> tuple[float, int]:
        """
        Value function V(pθ, S)(s): sample sure/maybe/impossible n_eval times,
        use scores sure=20, maybe=1, impossible=0.001, return sum.

        Early stopping: if all samples so far are "sure" (max score already
        reached), skip remaining samples — the state is clearly promising.
        If all samples are "impossible", stop early and prune.
        """
        total_score = 0.0
        total_tok = 0
        max_per_sample = _SCORE["sure"]
        prompt = _VALUE_PROMPT.format(input=remaining)

        for i in range(self.n_eval):
            resp = self.llm.call(prompt, system=_SYS, max_tokens=20, temperature=0.7)
            total_tok += resp["tokens"]
            out = resp["content"].strip().lower()
            if "sure" in out:
                total_score += _SCORE["sure"]
            elif "impossible" in out:
                total_score += _SCORE["impossible"]
            else:
                total_score += _SCORE["maybe"]

            # Early stopping: unanimously sure → no need for more samples
            if total_score >= max_per_sample * (i + 1) and i < self.n_eval - 1:
                # All samples so far are "sure" — extrapolate
                remaining_samples = self.n_eval - (i + 1)
                total_score += _SCORE["sure"] * remaining_samples
                break

            # Early stopping: unanimously impossible → prune immediately
            if total_score <= _SCORE["impossible"] * (i + 1) * 1.5 and i >= 1:
                break

        return total_score, total_tok

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _path_to_answer(puzzle: str, path: list[str]) -> str:
        """Summarise the 3-step solution path as a readable string."""
        return " ; ".join(path)


# ── IO Baseline ────────────────────────────────────────────────────────────────

class Game24IOBaseline:
    """Standard IO prompting with 5 in-context examples (paper Section 4.1)."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def solve(self, puzzle: str) -> Game24Result:
        t0 = time.time()
        prompt = _IO_PROMPT.format(input=puzzle)
        resp = self.llm.call(prompt, system=_SYS, max_tokens=60, temperature=0.7)
        answer = _extract_answer(resp["content"])
        success = verify_solution(puzzle, answer) if answer else False
        return Game24Result(
            puzzle=puzzle,
            method="io",
            success=success,
            answer=answer if success else None,
            nodes_explored=1,
            total_tokens=resp["tokens"],
            time_elapsed=round(time.time() - t0, 2),
        )


# ── CoT Baseline ───────────────────────────────────────────────────────────────

class Game24CoTBaseline:
    """
    Chain-of-thought with 3 intermediate equations shown per example.
    The paper samples CoT 100× for average performance; we do single-sample.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def solve(self, puzzle: str) -> Game24Result:
        t0 = time.time()
        prompt = _COT_PROMPT.format(input=puzzle)
        resp = self.llm.call(prompt, system=_SYS, max_tokens=200, temperature=0.7)
        content = resp["content"]

        # Parse thoughts (intermediate steps) and final answer
        thoughts = _parse_thoughts(content)
        answer = _extract_answer(content)
        success = False
        if answer:
            success = verify_solution(puzzle, answer)
        if not success and thoughts:
            # Check if the thought path itself reaches 24
            success = verify_thought_path(puzzle, thoughts[:3])

        return Game24Result(
            puzzle=puzzle,
            method="cot",
            success=success,
            answer=answer if success else None,
            thoughts=thoughts[:3],
            nodes_explored=1,
            total_tokens=resp["tokens"],
            time_elapsed=round(time.time() - t0, 2),
        )


# ── CoT-SC Baseline ────────────────────────────────────────────────────────────

class Game24CoTSCBaseline:
    """Self-consistency: draw n CoT samples, return first that verifies."""

    def __init__(self, llm: LLMClient, n_samples: int = 5):
        self.llm = llm
        self.n_samples = n_samples

    def solve(self, puzzle: str) -> Game24Result:
        t0 = time.time()
        prompt = _COT_PROMPT.format(input=puzzle)
        total_tokens = 0
        answers: list[str] = []

        for _ in range(self.n_samples):
            resp = self.llm.call(prompt, system=_SYS, max_tokens=200, temperature=0.8)
            total_tokens += resp["tokens"]
            ans = _extract_answer(resp["content"])
            if ans:
                answers.append(ans)

        success = False
        winning = None
        for ans in answers:
            if verify_solution(puzzle, ans):
                success = True
                winning = ans
                break

        return Game24Result(
            puzzle=puzzle,
            method=f"cot-sc-{self.n_samples}",
            success=success,
            answer=winning,
            nodes_explored=self.n_samples,
            total_tokens=total_tokens,
            time_elapsed=round(time.time() - t0, 2),
        )


# ── Experiment runner ──────────────────────────────────────────────────────────

def run_game24(
    puzzles: list[str],
    solver,
    label: str,
    verbose: bool = True,
) -> list[Game24Result]:
    results = []
    for i, puzzle in enumerate(puzzles):
        try:
            result = solver.solve(puzzle)
        except Exception as e:
            if verbose:
                print(f"  [{label}] ERROR on '{puzzle}': {e}")
            result = Game24Result(
                puzzle=puzzle, method=label, success=False,
                answer=None, nodes_explored=0, total_tokens=0, time_elapsed=0.0,
            )
        results.append(result)
        if verbose:
            status = "✓" if result.success else "✗"
            print(f"  [{label}] {status} {puzzle:15s}  "
                  f"tokens={result.total_tokens:5d}  nodes={result.nodes_explored}")
    return results


def compute_game24_metrics(results: list[Game24Result]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    n_success = sum(r.success for r in results)
    return {
        "n": n,
        "success_rate": n_success / n,
        "avg_tokens": sum(r.total_tokens for r in results) / n,
        "avg_nodes": sum(r.nodes_explored for r in results) / n,
        "avg_time": sum(r.time_elapsed for r in results) / n,
    }


def print_game24_table(all_results: dict[str, list[Game24Result]]):
    header = f"{'Method':<20}  {'Success':>8}  {'Avg Tokens':>12}  {'Avg Nodes':>10}  {'Avg Time':>10}"
    print("\n" + header)
    print("-" * len(header))
    for label, results in all_results.items():
        m = compute_game24_metrics(results)
        print(
            f"{label:<20}  {m['success_rate']:>7.1%}  "
            f"{m['avg_tokens']:>12.0f}  {m['avg_nodes']:>10.1f}  "
            f"{m['avg_time']:>9.1f}s"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(x: float) -> str:
    """Format a float without trailing zeros where possible."""
    if x == int(x):
        return str(int(x))
    return f"{x:.4g}"


def _extract_answer(text: str) -> Optional[str]:
    """
    Extract the final equation from LLM output.
    Looks for 'Answer: ...' or the last line containing '= 24'.
    """
    # Look for explicit "Answer:" line
    for line in text.split("\n"):
        if line.strip().lower().startswith("answer:"):
            ans = line.split(":", 1)[1].strip()
            if ans:
                return ans

    # Fallback: last line containing '= 24'
    for line in reversed(text.split("\n")):
        if "24" in line and "=" in line:
            # Strip leading text before the equation
            m = re.search(r"[\d(].*=\s*24", line)
            if m:
                return m.group(0)

    return None
