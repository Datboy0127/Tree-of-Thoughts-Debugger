"""
Tree of Thoughts (ToT) Debugger
================================
Implements BFS, DFS, and MCTS search over a two-level thought tree:
  Level 1 (depth=1): Bug hypotheses
  Level 2 (depth=2): Concrete code fixes

Search strategy
  BFS:  Evaluate all k hypotheses, select top-b, then expand all their fix candidates.
  DFS:  Commit greedily to the best hypothesis; backtrack if all its fixes fail.
  MCTS: Monte Carlo Tree Search — uses UCB1 to select hypotheses, generates a fix
        as the rollout, uses test pass/fail as reward, backpropagates through the tree.

References
  Yao et al., "Tree of Thoughts: Deliberate Problem Solving with LLMs", NeurIPS 2023.
  Coulom, "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search", 2006.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from llm_client import LLMClient
from executor import CodeExecutor
from data_loader import Problem


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    content: str
    depth: int
    node_type: str = "hypothesis"   # "hypothesis" | "fix"
    parent: Optional["ThoughtNode"] = None
    children: list = field(default_factory=list)
    score: float = 0.0
    fix_code: Optional[str] = None
    test_result: Optional[dict] = None
    tokens_used: int = 0


@dataclass
class MCTSNode:
    """A node in the MCTS tree representing one bug hypothesis."""
    hypothesis: str
    parent: Optional["MCTSNode"] = None
    children: list = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0          # cumulative reward (each rollout gives 0 or 1)
    fix_code: Optional[str] = None
    tokens_used: int = 0

    def ucb1(self, exploration: float = math.sqrt(2)) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        parent_visits = self.parent.visits if self.parent else self.visits
        exploration_term = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration_term

    def best_child(self, exploration: float = math.sqrt(2)) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class DebugResult:
    task_id: str
    method: str
    success: bool
    fix_code: Optional[str]
    nodes_explored: int
    backtracks: int
    total_tokens: int
    time_elapsed: float
    first_attempt_success: bool
    bug_type: str = "unknown"


# ── Prompt templates ───────────────────────────────────────────────────────────

_HYPOTHESIS_PROMPT = """\
You are an expert Python debugger. Analyze the buggy function below and propose {k} \
distinct, numbered hypotheses about what is causing the bug.

### Problem description
{prompt}

### Buggy code
```python
{buggy_code}
```

For each hypothesis use this exact format:

HYPOTHESIS 1: [short title]
EXPLANATION: [why this causes the observed failures]
LOCATION: [line number, variable, or expression]

HYPOTHESIS 2: ...

Generate exactly {k} hypotheses:"""

_EVAL_PROMPT = """\
Rate the following debugging hypothesis from 1 to 10 (10 = almost certainly correct).

### Buggy code
```python
{buggy_code}
```

### Hypothesis
{hypothesis}

Reply with a single integer (1–10) and nothing else:"""

_FIX_PROMPT = """\
You are an expert Python programmer. Given the bug diagnosis below, generate {k} \
distinct complete Python function implementations that fix the bug.

### Problem description
{prompt}

### Buggy code
```python
{buggy_code}
```

### Bug diagnosis
{hypothesis}

Format each fix exactly as:

FIX 1:
```python
[complete function]
```

FIX 2:
```python
[complete function]
```

Generate exactly {k} fixes:"""


# ── Main class ────────────────────────────────────────────────────────────────

class ToTDebugger:
    """
    Tree of Thoughts debugger.

    Parameters
    ----------
    llm            : LLMClient
    executor       : CodeExecutor
    k              : branching factor (candidates per level)
    search         : "bfs" | "dfs" | "mcts"
    evaluator      : "llm" | "execution" | "hybrid"
    n_simulations  : MCTS iterations per problem (default 12)
    exploration    : MCTS UCB1 exploration constant C (default √2)
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: CodeExecutor,
        k: int = 3,
        search: str = "bfs",
        evaluator: str = "hybrid",
        n_simulations: int = 12,
        exploration: float = math.sqrt(2),
    ):
        self.llm = llm
        self.executor = executor
        self.k = k
        self.search = search
        self.evaluator = evaluator
        self.n_simulations = n_simulations
        self.exploration = exploration

    # ── Public API ─────────────────────────────────────────────────────────────

    def solve(self, problem: Problem) -> DebugResult:
        t0 = time.time()
        if self.search == "bfs":
            result = self._bfs_solve(problem)
        elif self.search == "dfs":
            result = self._dfs_solve(problem)
        else:
            result = self._mcts_solve(problem)
        result.time_elapsed = round(time.time() - t0, 2)
        result.bug_type = problem.bug_type
        return result

    # ── BFS ───────────────────────────────────────────────────────────────────

    def _bfs_solve(self, problem: Problem) -> DebugResult:
        nodes_explored = 0
        total_tokens = 0

        # Level 1 – hypotheses
        hypotheses = self._generate_hypotheses(problem)
        total_tokens += sum(h.tokens_used for h in hypotheses)
        nodes_explored += len(hypotheses)

        for h in hypotheses:
            score, tok = self._score_hypothesis(h, problem)
            h.score = score
            total_tokens += tok

        hypotheses.sort(key=lambda x: x.score, reverse=True)
        top = hypotheses[: self.k]

        # Level 2 – fixes for all top hypotheses (BFS breadth)
        first_node = True
        for hypothesis in top:
            fixes = self._generate_fixes(hypothesis, problem)
            hypothesis.children = fixes
            total_tokens += sum(f.tokens_used for f in fixes)
            nodes_explored += len(fixes)

            for fix_node in fixes:
                result = self.executor.execute(fix_node.fix_code, problem.test_code)
                fix_node.test_result = result
                if result["passed"]:
                    return DebugResult(
                        task_id=problem.task_id,
                        method=f"tot-bfs-k{self.k}",
                        success=True,
                        fix_code=fix_node.fix_code,
                        nodes_explored=nodes_explored,
                        backtracks=0,
                        total_tokens=total_tokens,
                        time_elapsed=0,
                        first_attempt_success=first_node,
                    )
                first_node = False

        return DebugResult(
            task_id=problem.task_id,
            method=f"tot-bfs-k{self.k}",
            success=False,
            fix_code=None,
            nodes_explored=nodes_explored,
            backtracks=0,
            total_tokens=total_tokens,
            time_elapsed=0,
            first_attempt_success=False,
        )

    # ── DFS ───────────────────────────────────────────────────────────────────

    def _dfs_solve(self, problem: Problem) -> DebugResult:
        nodes_explored = 0
        total_tokens = 0
        backtracks = 0
        first_overall = True

        hypotheses = self._generate_hypotheses(problem)
        total_tokens += sum(h.tokens_used for h in hypotheses)
        nodes_explored += len(hypotheses)

        for h in hypotheses:
            score, tok = self._score_hypothesis(h, problem)
            h.score = score
            total_tokens += tok

        hypotheses.sort(key=lambda x: x.score, reverse=True)

        for hypothesis in hypotheses:
            fixes = self._generate_fixes(hypothesis, problem)
            hypothesis.children = fixes
            total_tokens += sum(f.tokens_used for f in fixes)
            nodes_explored += len(fixes)

            for i, fix_node in enumerate(fixes):
                result = self.executor.execute(fix_node.fix_code, problem.test_code)
                fix_node.test_result = result
                if result["passed"]:
                    return DebugResult(
                        task_id=problem.task_id,
                        method=f"tot-dfs-k{self.k}",
                        success=True,
                        fix_code=fix_node.fix_code,
                        nodes_explored=nodes_explored,
                        backtracks=backtracks,
                        total_tokens=total_tokens,
                        time_elapsed=0,
                        first_attempt_success=(first_overall and i == 0),
                    )
                first_overall = False

            # All fixes for this hypothesis failed → backtrack
            backtracks += 1

        return DebugResult(
            task_id=problem.task_id,
            method=f"tot-dfs-k{self.k}",
            success=False,
            fix_code=None,
            nodes_explored=nodes_explored,
            backtracks=backtracks,
            total_tokens=total_tokens,
            time_elapsed=0,
            first_attempt_success=False,
        )

    # ── MCTS ──────────────────────────────────────────────────────────────────

    def _mcts_solve(self, problem: Problem) -> DebugResult:
        """
        Monte Carlo Tree Search over bug hypotheses.

        Tree structure:
          root (virtual) → MCTSNode per hypothesis → rollout (generate fix + execute)

        Each iteration:
          1. Selection   — walk tree via UCB1 to find the most promising leaf
          2. Expansion   — if the leaf has no children, generate k new hypotheses
          3. Simulation  — generate one fix for the selected node, run tests (reward 0/1)
          4. Backprop    — update visits and wins up to root
        """
        total_tokens = 0
        nodes_explored = 0
        best_fix: Optional[str] = None
        first_attempt_success = False

        # Root holds the initial hypothesis pool (expanded once)
        root = MCTSNode(hypothesis="root")
        self._last_mcts_root = root  # expose for visualization

        # Seed root with k initial hypotheses
        hypotheses = self._generate_hypotheses(problem)
        total_tokens += sum(h.tokens_used for h in hypotheses)
        for h in hypotheses:
            child = MCTSNode(hypothesis=h.content, parent=root, tokens_used=h.tokens_used)
            root.children.append(child)
        nodes_explored += len(root.children)

        for sim in range(self.n_simulations):
            # ── 1. Selection ───────────────────────────────────────────────
            node = root
            while not node.is_leaf() and node.visits > 0:
                node = node.best_child(self.exploration)

            # ── 2. Expansion ───────────────────────────────────────────────
            # If this node has been visited before, expand with a new hypothesis
            if node.visits > 0 and node is not root:
                new_hyps = self._generate_hypotheses(problem)
                total_tokens += sum(h.tokens_used for h in new_hyps)
                for h in new_hyps:
                    already = any(c.hypothesis == h.content for c in root.children)
                    if not already:
                        child = MCTSNode(hypothesis=h.content, parent=root, tokens_used=h.tokens_used)
                        root.children.append(child)
                        nodes_explored += 1
                # Re-select after expansion
                node = root.best_child(self.exploration)

            # ── 3. Simulation (rollout) ────────────────────────────────────
            # Generate one fix for this hypothesis and run the tests
            thought = ThoughtNode(content=node.hypothesis, depth=1, node_type="hypothesis")
            fixes = self._generate_fixes(thought, problem)
            total_tokens += sum(f.tokens_used for f in fixes)
            nodes_explored += len(fixes)

            reward = 0.0
            for fix_node in fixes:
                result = self.executor.execute(fix_node.fix_code, problem.test_code)
                fix_node.test_result = result
                if result["passed"]:
                    reward = 1.0
                    if best_fix is None:
                        best_fix = fix_node.fix_code
                        first_attempt_success = (sim == 0)
                    break
                # Partial reward: fraction of passing tests (encourages progress)
                elif result["stderr"] == "" and result["stdout"] != "":
                    reward = max(reward, 0.1)

            node.fix_code = fixes[0].fix_code if fixes else None

            # ── 4. Backpropagation ─────────────────────────────────────────
            current = node
            while current is not None:
                current.visits += 1
                current.wins += reward
                current = current.parent

            if best_fix is not None and reward == 1.0:
                # Early exit if we already found a passing fix
                # (continue remaining simulations to improve hypothesis ranking stats)
                pass

        success = best_fix is not None
        return DebugResult(
            task_id=problem.task_id,
            method=f"mcts-k{self.k}-s{self.n_simulations}",
            success=success,
            fix_code=best_fix,
            nodes_explored=nodes_explored,
            backtracks=0,
            total_tokens=total_tokens,
            time_elapsed=0,
            first_attempt_success=first_attempt_success,
        )

    # ── Thought generation ────────────────────────────────────────────────────

    def _generate_hypotheses(self, problem: Problem) -> list[ThoughtNode]:
        prompt = _HYPOTHESIS_PROMPT.format(
            k=self.k,
            prompt=problem.prompt.strip(),
            buggy_code=problem.buggy_code.strip(),
        )
        resp = self.llm.call(prompt)
        parsed = _parse_hypotheses(resp["content"])
        if not parsed:
            parsed = [f"Generic fix attempt {i+1}" for i in range(self.k)]
        return [
            ThoughtNode(content=h, depth=1, node_type="hypothesis", tokens_used=resp["tokens"] // max(len(parsed), 1))
            for h in parsed[: self.k]
        ]

    def _generate_fixes(self, hypothesis: ThoughtNode, problem: Problem) -> list[ThoughtNode]:
        prompt = _FIX_PROMPT.format(
            k=self.k,
            prompt=problem.prompt.strip(),
            buggy_code=problem.buggy_code.strip(),
            hypothesis=hypothesis.content.strip(),
        )
        resp = self.llm.call(prompt)
        codes = _parse_fixes(resp["content"])
        if not codes:
            codes = [problem.buggy_code]
        return [
            ThoughtNode(
                content=f"Fix for: {hypothesis.content[:60]}",
                depth=2,
                node_type="fix",
                parent=hypothesis,
                fix_code=c,
                tokens_used=resp["tokens"] // max(len(codes), 1),
            )
            for c in codes[: self.k]
        ]

    # ── Hypothesis evaluation ─────────────────────────────────────────────────

    def _score_hypothesis(self, node: ThoughtNode, problem: Problem) -> tuple[float, int]:
        if self.evaluator == "execution":
            return self._execution_score(node, problem)
        if self.evaluator == "hybrid":
            llm_score, tok = self._llm_score(node, problem)
            exec_score, _ = self._execution_score(node, problem)
            return 0.6 * llm_score + 0.4 * exec_score, tok
        return self._llm_score(node, problem)

    def _llm_score(self, node: ThoughtNode, problem: Problem) -> tuple[float, int]:
        prompt = _EVAL_PROMPT.format(
            buggy_code=problem.buggy_code.strip(),
            hypothesis=node.content.strip(),
        )
        resp = self.llm.call(prompt, max_tokens=10)
        score = _parse_score(resp["content"])
        return score / 10.0, resp["tokens"]

    def _execution_score(self, node: ThoughtNode, problem: Problem) -> tuple[float, int]:
        # Ask LLM for a quick fix and test it
        quick_prompt = (
            f"Given this bug hypothesis:\n{node.content}\n\n"
            f"Fix this code in one attempt:\n```python\n{problem.buggy_code}\n```\n"
            "Reply with only the complete fixed function inside ```python``` fences:"
        )
        resp = self.llm.call(quick_prompt, max_tokens=400)
        codes = _parse_fixes(resp["content"])
        if codes:
            result = self.executor.execute(codes[0], problem.test_code)
            score = 1.0 if result["passed"] else 0.2
        else:
            score = 0.1
        return score, resp["tokens"]


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_hypotheses(text: str) -> list[str]:
    blocks = re.split(r"\n?HYPOTHESIS\s+\d+\s*:", text, flags=re.IGNORECASE)
    results = []
    for block in blocks[1:]:
        block = block.strip()
        # Collapse whitespace between sections
        block = re.sub(r"\n(EXPLANATION|LOCATION)\s*:", r" \1:", block, flags=re.IGNORECASE)
        if block:
            results.append(block.strip())
    # Fallback: split numbered list
    if not results:
        for m in re.finditer(r"^\d+\.\s+(.+?)(?=\n\d+\.|\Z)", text, re.DOTALL | re.MULTILINE):
            results.append(m.group(1).strip())
    return results


def _parse_fixes(text: str) -> list[str]:
    # Extract ```python ... ``` blocks
    codes = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if codes:
        return [c.strip() for c in codes if c.strip()]
    # Fallback: extract FIX N: sections
    blocks = re.split(r"\nFIX\s+\d+\s*:", text, flags=re.IGNORECASE)
    results = []
    for block in blocks[1:]:
        code = re.sub(r"(?i)FIX\s+\d+.*", "", block).strip()
        if code:
            results.append(code)
    return results


def _parse_score(text: str) -> float:
    match = re.search(r"\b([1-9]|10)\b", text.strip())
    if match:
        return float(match.group(1))
    return 5.0
