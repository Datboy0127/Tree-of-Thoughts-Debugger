"""
Tree of Thoughts (ToT) Debugger
================================
3-level thought tree matching Yao et al. (NeurIPS 2023) structure:

  Level 1 (Area):       Which component of the code contains the bug?
  Level 2 (Hypothesis): What specifically is wrong with that component?
  Level 3 (Fix):        Concrete corrected code, verified by test execution.

This mirrors the paper's 3-step Game of 24 tree (depth=3, branching=5).

Evaluation uses the paper's exact scheme: the LLM rates each thought as
"sure / likely / impossible", and impossible states are pruned immediately.

Search strategies
  BFS:  Generate k candidates at each level, prune impossible, keep top-k,
        expand all survivors before moving to the next level.
  DFS:  Greedy descent; backtrack at hypothesis level first, then area level.
        "Impossible" thoughts are pruned without expansion.
  MCTS: UCB1-guided selection over hypotheses; test execution as reward;
        backpropagation corrects noisy initial LLM scores over rollouts.

References
  Yao et al., "Tree of Thoughts", NeurIPS 2023.
  Coulom, "Efficient Selectivity and Backup Operators in MCTS", CG 2006.
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
    depth: int                          # 1=area, 2=hypothesis, 3=fix
    node_type: str = "area"             # "area" | "hypothesis" | "fix"
    parent: Optional["ThoughtNode"] = None
    children: list = field(default_factory=list)
    score: float = 0.5                  # 0=impossible, 0.5=likely, 1.0=sure
    fix_code: Optional[str] = None
    test_result: Optional[dict] = None
    tokens_used: int = 0


@dataclass
class MCTSNode:
    hypothesis: str
    parent: Optional["MCTSNode"] = None
    children: list = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0
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


# ── Prompt templates (matching paper's evaluation scheme) ──────────────────────

# Level 1: Identify which code area is buggy
_AREA_PROMPT = """\
You are an expert Python debugger. Given the buggy function below, identify \
the {k} most suspicious code areas that could contain the bug.

### Buggy code
```python
{buggy_code}
```

### Failing tests hint
{prompt}

For each area use exactly this format:

AREA 1: [short name, e.g. "loop bounds", "return value", "condition check"]
DESCRIPTION: [what this area does]
SUSPICION: [why it might be causing the bug]

AREA 2: ...

Generate exactly {k} areas:"""

# Level 1 evaluation: sure / likely / impossible (exactly as in the paper)
_AREA_EVAL_PROMPT = """\
Given this buggy Python code:

```python
{buggy_code}
```

Is the following code area likely to contain the bug?

Area: {area}

Reply with exactly one word — sure, likely, or impossible:"""

# Level 2: Generate specific hypotheses given a code area
_HYPOTHESIS_PROMPT = """\
You are an expert Python debugger. Given the buggy function and the suspected \
buggy area, generate {k} specific hypotheses about what exactly is wrong.

### Buggy code
```python
{buggy_code}
```

### Suspected buggy area
{area}

For each hypothesis use exactly this format:

HYPOTHESIS 1: [one sentence — what is wrong, e.g. "loop uses < instead of <="]
LOCATION: [line or expression]
IMPACT: [how this causes the test to fail]

HYPOTHESIS 2: ...

Generate exactly {k} hypotheses:"""

# Level 2 evaluation: sure / likely / impossible
_HYPOTHESIS_EVAL_PROMPT = """\
Given this buggy Python code:

```python
{buggy_code}
```

Does the following hypothesis correctly identify the bug?

Hypothesis: {hypothesis}

Reply with exactly one word — sure, likely, or impossible:"""

# Level 3: Generate concrete fixes
_FIX_PROMPT = """\
You are an expert Python programmer. Given the bug diagnosis below, write \
{k} distinct complete Python implementations that fix the bug.

### Buggy code
```python
{buggy_code}
```

### Bug area
{area}

### Specific diagnosis
{hypothesis}

Format each fix as:

FIX 1:
```python
[complete fixed function]
```

FIX 2:
```python
[complete fixed function]
```

Generate exactly {k} fixes:"""


# Score mapping: paper uses sure/likely/impossible
_SCORE_MAP = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}


# ── Main class ────────────────────────────────────────────────────────────────

class ToTDebugger:
    """
    Tree of Thoughts debugger with depth-3 thought tree.

    Parameters
    ----------
    llm            : LLMClient
    executor       : CodeExecutor
    k              : branching factor (paper uses b=5)
    search         : "bfs" | "dfs" | "mcts"
    evaluator      : "llm" (sure/likely/impossible, matches paper) |
                     "hybrid" | "execution"
    n_simulations  : MCTS rollouts (default 12)
    exploration    : MCTS UCB1 constant C (default √2)
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: CodeExecutor,
        k: int = 5,
        search: str = "bfs",
        evaluator: str = "llm",
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

    # ── BFS (depth-3, matches paper) ──────────────────────────────────────────

    def _bfs_solve(self, problem: Problem) -> DebugResult:
        """
        BFS over 3-level tree.
        Level 1: Generate k areas, evaluate all, prune impossible, keep top-k.
        Level 2: For each surviving area, generate k hypotheses, evaluate,
                 prune impossible, keep top-k across all areas combined.
        Level 3: For each surviving hypothesis, generate k fixes, execute.
        First passing fix is returned immediately.
        """
        nodes_explored = 0
        total_tokens = 0
        first_node = True

        # ── Level 1: Areas ────────────────────────────────────────────────────
        areas, tok = self._generate_areas(problem)
        total_tokens += tok
        nodes_explored += len(areas)

        for area in areas:
            score, tok = self._score_area(area, problem)
            area.score = score
            total_tokens += tok

        # Prune impossible areas (score == 0), keep top-k survivors
        areas = [a for a in areas if a.score > 0]
        areas.sort(key=lambda x: x.score, reverse=True)
        top_areas = areas[: self.k]

        if not top_areas:
            return self._failure(problem, "tot-bfs", nodes_explored, 0, total_tokens)

        # ── Level 2: Hypotheses ───────────────────────────────────────────────
        all_hypotheses: list[ThoughtNode] = []
        for area in top_areas:
            hyps, tok = self._generate_hypotheses(area, problem)
            total_tokens += tok
            nodes_explored += len(hyps)
            for h in hyps:
                score, tok2 = self._score_hypothesis(h, area, problem)
                h.score = score
                total_tokens += tok2
            area.children = hyps
            all_hypotheses.extend(hyps)

        # Prune impossible hypotheses, keep global top-k
        all_hypotheses = [h for h in all_hypotheses if h.score > 0]
        all_hypotheses.sort(key=lambda x: x.score, reverse=True)
        top_hypotheses = all_hypotheses[: self.k]

        if not top_hypotheses:
            return self._failure(problem, "tot-bfs", nodes_explored, 0, total_tokens)

        # ── Level 3: Fixes + execution ────────────────────────────────────────
        for hypothesis in top_hypotheses:
            area_node = hypothesis.parent
            fixes, tok = self._generate_fixes(area_node, hypothesis, problem)
            total_tokens += tok
            nodes_explored += len(fixes)
            hypothesis.children = fixes

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

        return self._failure(problem, f"tot-bfs-k{self.k}", nodes_explored, 0, total_tokens)

    # ── DFS (depth-3, 2-level backtracking) ───────────────────────────────────

    def _dfs_solve(self, problem: Problem) -> DebugResult:
        """
        DFS over 3-level tree with 2-level backtracking.
        Greedy: pick best area → best hypothesis → try all k fixes.
        Backtrack order: hypothesis level first, then area level.
        Impossible states are pruned without expansion (matches paper).
        """
        nodes_explored = 0
        total_tokens = 0
        backtracks = 0
        first_overall = True

        # ── Level 1: Areas ────────────────────────────────────────────────────
        areas, tok = self._generate_areas(problem)
        total_tokens += tok
        nodes_explored += len(areas)

        for area in areas:
            score, tok = self._score_area(area, problem)
            area.score = score
            total_tokens += tok

        # Prune impossible, sort by score descending
        areas = [a for a in areas if a.score > 0]
        areas.sort(key=lambda x: x.score, reverse=True)

        for area in areas:
            # ── Level 2: Hypotheses for this area ─────────────────────────────
            hyps, tok = self._generate_hypotheses(area, problem)
            total_tokens += tok
            nodes_explored += len(hyps)

            for h in hyps:
                score, tok2 = self._score_hypothesis(h, area, problem)
                h.score = score
                total_tokens += tok2

            hyps = [h for h in hyps if h.score > 0]
            hyps.sort(key=lambda x: x.score, reverse=True)
            area.children = hyps

            for hypothesis in hyps:
                # ── Level 3: Fixes ─────────────────────────────────────────────
                fixes, tok = self._generate_fixes(area, hypothesis, problem)
                total_tokens += tok
                nodes_explored += len(fixes)
                hypothesis.children = fixes

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

                # All fixes for this hypothesis failed → backtrack to next hypothesis
                backtracks += 1

            # All hypotheses for this area failed → backtrack to next area
            backtracks += 1

        return self._failure(problem, f"tot-dfs-k{self.k}", nodes_explored, backtracks, total_tokens)

    # ── MCTS ──────────────────────────────────────────────────────────────────

    def _mcts_solve(self, problem: Problem) -> DebugResult:
        """
        MCTS over hypothesis nodes with test execution as reward.
        Operates at the hypothesis level (level 2); areas are generated
        once to seed the hypothesis pool.
        UCB1 allocates more rollouts to promising hypotheses over time.
        """
        total_tokens = 0
        nodes_explored = 0
        best_fix: Optional[str] = None
        first_attempt_success = False

        root = MCTSNode(hypothesis="root")
        self._last_mcts_root = root

        # Seed root: generate areas then hypotheses from each area
        areas, tok = self._generate_areas(problem)
        total_tokens += tok
        for area in areas:
            hyps, tok2 = self._generate_hypotheses(area, problem)
            total_tokens += tok2
            for h in hyps:
                child = MCTSNode(
                    hypothesis=f"[{area.content[:30]}] {h.content}",
                    parent=root,
                    tokens_used=h.tokens_used,
                )
                root.children.append(child)
        nodes_explored += len(root.children)

        for sim in range(self.n_simulations):
            # ── 1. Selection ───────────────────────────────────────────────
            node = root
            while not node.is_leaf() and node.visits > 0:
                node = node.best_child(self.exploration)

            # ── 2. Expansion ───────────────────────────────────────────────
            if node.visits > 0 and node is not root:
                new_areas, tok = self._generate_areas(problem)
                total_tokens += tok
                for area in new_areas[:1]:   # one new area per expansion
                    new_hyps, tok2 = self._generate_hypotheses(area, problem)
                    total_tokens += tok2
                    for h in new_hyps[:self.k]:
                        label = f"[{area.content[:30]}] {h.content}"
                        if not any(c.hypothesis == label for c in root.children):
                            child = MCTSNode(hypothesis=label, parent=root, tokens_used=h.tokens_used)
                            root.children.append(child)
                            nodes_explored += 1
                node = root.best_child(self.exploration)

            # ── 3. Simulation (rollout) ────────────────────────────────────
            thought = ThoughtNode(content=node.hypothesis, depth=2, node_type="hypothesis")
            area_dummy = ThoughtNode(content="", depth=1, node_type="area")
            fixes, tok = self._generate_fixes(area_dummy, thought, problem)
            total_tokens += tok
            nodes_explored += len(fixes)

            reward = 0.0
            for fix_node in fixes:
                result = self.executor.execute(fix_node.fix_code, problem.test_code)
                if result["passed"]:
                    reward = 1.0
                    if best_fix is None:
                        best_fix = fix_node.fix_code
                        first_attempt_success = (sim == 0)
                    break
                elif not result.get("stderr", ""):
                    reward = max(reward, 0.1)

            node.fix_code = fixes[0].fix_code if fixes else None

            # ── 4. Backpropagation ─────────────────────────────────────────
            current = node
            while current is not None:
                current.visits += 1
                current.wins += reward
                current = current.parent

        return DebugResult(
            task_id=problem.task_id,
            method=f"mcts-k{self.k}-s{self.n_simulations}",
            success=best_fix is not None,
            fix_code=best_fix,
            nodes_explored=nodes_explored,
            backtracks=0,
            total_tokens=total_tokens,
            time_elapsed=0,
            first_attempt_success=first_attempt_success,
        )

    # ── Level 1: Area generation and evaluation ────────────────────────────────

    def _generate_areas(self, problem: Problem) -> tuple[list[ThoughtNode], int]:
        prompt = _AREA_PROMPT.format(
            k=self.k,
            buggy_code=problem.buggy_code.strip(),
            prompt=problem.prompt.strip()[:300],
        )
        resp = self.llm.call(prompt, max_tokens=600)
        parsed = _parse_areas(resp["content"])
        if not parsed:
            parsed = [f"Area {i+1}: general code logic" for i in range(self.k)]
        nodes = [
            ThoughtNode(content=a, depth=1, node_type="area",
                        tokens_used=resp["tokens"] // max(len(parsed), 1))
            for a in parsed[: self.k]
        ]
        return nodes, resp["tokens"]

    def _score_area(self, area: ThoughtNode, problem: Problem) -> tuple[float, int]:
        if self.evaluator != "llm":
            return 0.5, 0   # neutral score for non-LLM evaluators
        prompt = _AREA_EVAL_PROMPT.format(
            buggy_code=problem.buggy_code.strip(),
            area=area.content.strip(),
        )
        resp = self.llm.call(prompt, max_tokens=10)
        score = _parse_sure_likely_impossible(resp["content"])
        return score, resp["tokens"]

    # ── Level 2: Hypothesis generation and evaluation ──────────────────────────

    def _generate_hypotheses(self, area: ThoughtNode, problem: Problem) -> tuple[list[ThoughtNode], int]:
        prompt = _HYPOTHESIS_PROMPT.format(
            k=self.k,
            buggy_code=problem.buggy_code.strip(),
            area=area.content.strip() if area.content else "the code logic",
        )
        resp = self.llm.call(prompt, max_tokens=800)
        parsed = _parse_hypotheses(resp["content"])
        if not parsed:
            parsed = [f"Generic hypothesis {i+1}" for i in range(self.k)]
        nodes = [
            ThoughtNode(content=h, depth=2, node_type="hypothesis",
                        parent=area,
                        tokens_used=resp["tokens"] // max(len(parsed), 1))
            for h in parsed[: self.k]
        ]
        return nodes, resp["tokens"]

    def _score_hypothesis(self, node: ThoughtNode, area: ThoughtNode, problem: Problem) -> tuple[float, int]:
        if self.evaluator == "execution":
            return self._execution_score(node, problem)
        if self.evaluator == "hybrid":
            llm_score, tok = self._llm_score_hypothesis(node, problem)
            exec_score, _ = self._execution_score(node, problem)
            return 0.6 * llm_score + 0.4 * exec_score, tok
        return self._llm_score_hypothesis(node, problem)

    def _llm_score_hypothesis(self, node: ThoughtNode, problem: Problem) -> tuple[float, int]:
        prompt = _HYPOTHESIS_EVAL_PROMPT.format(
            buggy_code=problem.buggy_code.strip(),
            hypothesis=node.content.strip(),
        )
        resp = self.llm.call(prompt, max_tokens=10)
        score = _parse_sure_likely_impossible(resp["content"])
        return score, resp["tokens"]

    def _execution_score(self, node: ThoughtNode, problem: Problem) -> tuple[float, int]:
        quick_prompt = (
            f"Fix this bug based on the hypothesis below.\n"
            f"Hypothesis: {node.content}\n\n"
            f"Code:\n```python\n{problem.buggy_code}\n```\n"
            "Return only the fixed function in ```python``` fences:"
        )
        resp = self.llm.call(quick_prompt, max_tokens=400)
        codes = _parse_fixes(resp["content"])
        if codes:
            result = self.executor.execute(codes[0], problem.test_code)
            return (1.0 if result["passed"] else 0.2), resp["tokens"]
        return 0.1, resp["tokens"]

    # ── Level 3: Fix generation ────────────────────────────────────────────────

    def _generate_fixes(
        self,
        area: ThoughtNode,
        hypothesis: ThoughtNode,
        problem: Problem,
    ) -> tuple[list[ThoughtNode], int]:
        prompt = _FIX_PROMPT.format(
            k=self.k,
            buggy_code=problem.buggy_code.strip(),
            area=area.content.strip() if area.content else "unknown area",
            hypothesis=hypothesis.content.strip(),
        )
        resp = self.llm.call(prompt, max_tokens=800)
        codes = _parse_fixes(resp["content"])
        if not codes:
            codes = [problem.buggy_code]
        nodes = [
            ThoughtNode(
                content=f"Fix for: {hypothesis.content[:60]}",
                depth=3,
                node_type="fix",
                parent=hypothesis,
                fix_code=c,
                tokens_used=resp["tokens"] // max(len(codes), 1),
            )
            for c in codes[: self.k]
        ]
        return nodes, resp["tokens"]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _failure(self, problem: Problem, method: str, nodes: int, backtracks: int, tokens: int) -> DebugResult:
        return DebugResult(
            task_id=problem.task_id,
            method=method,
            success=False,
            fix_code=None,
            nodes_explored=nodes,
            backtracks=backtracks,
            total_tokens=tokens,
            time_elapsed=0,
            first_attempt_success=False,
        )


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_sure_likely_impossible(text: str) -> float:
    """Convert sure/likely/impossible to numeric score (matches paper's scheme)."""
    text = text.strip().lower()
    for word, score in _SCORE_MAP.items():
        if word in text:
            return score
    # Fallback: treat any response as likely
    return 0.5


def _parse_areas(text: str) -> list[str]:
    blocks = re.split(r"\n?AREA\s+\d+\s*:", text, flags=re.IGNORECASE)
    results = []
    for block in blocks[1:]:
        block = re.sub(r"\n(DESCRIPTION|SUSPICION)\s*:", r" \1:", block, flags=re.IGNORECASE)
        block = block.strip()
        if block:
            results.append(block.strip())
    if not results:
        for m in re.finditer(r"^\d+\.\s+(.+?)(?=\n\d+\.|\Z)", text, re.DOTALL | re.MULTILINE):
            results.append(m.group(1).strip())
    return results


def _parse_hypotheses(text: str) -> list[str]:
    blocks = re.split(r"\n?HYPOTHESIS\s+\d+\s*:", text, flags=re.IGNORECASE)
    results = []
    for block in blocks[1:]:
        block = re.sub(r"\n(LOCATION|IMPACT)\s*:", r" \1:", block, flags=re.IGNORECASE)
        block = block.strip()
        if block:
            results.append(block.strip())
    if not results:
        for m in re.finditer(r"^\d+\.\s+(.+?)(?=\n\d+\.|\Z)", text, re.DOTALL | re.MULTILINE):
            results.append(m.group(1).strip())
    return results


def _parse_fixes(text: str) -> list[str]:
    codes = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if codes:
        return [c.strip() for c in codes if c.strip()]
    blocks = re.split(r"\nFIX\s+\d+\s*:", text, flags=re.IGNORECASE)
    results = []
    for block in blocks[1:]:
        code = re.sub(r"(?i)FIX\s+\d+.*", "", block).strip()
        if code:
            results.append(code)
    return results


_SCORE_MAP = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}
