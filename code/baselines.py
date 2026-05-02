"""
Baseline methods for code debugging:
  IO  – direct input/output prompting
  CoT – chain-of-thought prompting
  CoT-SC – self-consistency (majority vote over multiple CoT samples)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional
from collections import Counter

from llm_client import LLMClient
from executor import CodeExecutor
from data_loader import Problem
from tot_debugger import DebugResult, _parse_fixes


# ── Prompt templates ───────────────────────────────────────────────────────────

_IO_PROMPT = """\
Fix the following buggy Python function. Return only the complete corrected function \
inside ```python``` fences.

### Problem description
{prompt}

### Buggy code
```python
{buggy_code}
```
"""

_COT_PROMPT = """\
Fix the following buggy Python function by reasoning step by step.

### Problem description
{prompt}

### Buggy code
```python
{buggy_code}
```

Let's think step by step about what is wrong:
1. What is this function supposed to do?
2. What bug(s) might cause it to fail?
3. What is the corrected code?

After your reasoning, provide the complete fixed function inside ```python``` fences:"""


# ── Baseline runners ──────────────────────────────────────────────────────────

class IOBaseline:
    """Direct prompting: one shot, no reasoning chain."""

    def __init__(self, llm: LLMClient, executor: CodeExecutor):
        self.llm = llm
        self.executor = executor

    def solve(self, problem: Problem) -> DebugResult:
        t0 = time.time()
        prompt = _IO_PROMPT.format(
            prompt=problem.prompt.strip(),
            buggy_code=problem.buggy_code.strip(),
        )
        resp = self.llm.call(prompt)
        codes = _parse_fixes(resp["content"])
        fix = codes[0] if codes else problem.buggy_code

        result = self.executor.execute(fix, problem.test_code)
        return DebugResult(
            task_id=problem.task_id,
            method="io",
            success=result["passed"],
            fix_code=fix if result["passed"] else None,
            nodes_explored=1,
            backtracks=0,
            total_tokens=resp["tokens"],
            time_elapsed=round(time.time() - t0, 2),
            first_attempt_success=result["passed"],
            bug_type=problem.bug_type,
        )


class CoTBaseline:
    """Chain-of-thought prompting: single sample."""

    def __init__(self, llm: LLMClient, executor: CodeExecutor):
        self.llm = llm
        self.executor = executor

    def solve(self, problem: Problem) -> DebugResult:
        t0 = time.time()
        prompt = _COT_PROMPT.format(
            prompt=problem.prompt.strip(),
            buggy_code=problem.buggy_code.strip(),
        )
        resp = self.llm.call(prompt, max_tokens=800)
        codes = _parse_fixes(resp["content"])
        fix = codes[0] if codes else problem.buggy_code

        result = self.executor.execute(fix, problem.test_code)
        return DebugResult(
            task_id=problem.task_id,
            method="cot",
            success=result["passed"],
            fix_code=fix if result["passed"] else None,
            nodes_explored=1,
            backtracks=0,
            total_tokens=resp["tokens"],
            time_elapsed=round(time.time() - t0, 2),
            first_attempt_success=result["passed"],
            bug_type=problem.bug_type,
        )


class CoTSCBaseline:
    """
    Self-consistency (Wang et al., 2022).
    Draw n CoT samples; select the most-voted fix by test-pass majority.
    """

    def __init__(self, llm: LLMClient, executor: CodeExecutor, n_samples: int = 5):
        self.llm = llm
        self.executor = executor
        self.n_samples = n_samples

    def solve(self, problem: Problem) -> DebugResult:
        t0 = time.time()
        prompt = _COT_PROMPT.format(
            prompt=problem.prompt.strip(),
            buggy_code=problem.buggy_code.strip(),
        )

        all_codes: list[str] = []
        total_tokens = 0
        for _ in range(self.n_samples):
            resp = self.llm.call(prompt, max_tokens=800, temperature=0.8)
            total_tokens += resp["tokens"]
            codes = _parse_fixes(resp["content"])
            if codes:
                all_codes.append(codes[0])

        # Majority vote by string equality (exact match)
        if not all_codes:
            return DebugResult(
                task_id=problem.task_id,
                method=f"cot-sc-{self.n_samples}",
                success=False,
                fix_code=None,
                nodes_explored=self.n_samples,
                backtracks=0,
                total_tokens=total_tokens,
                time_elapsed=round(time.time() - t0, 2),
                first_attempt_success=False,
                bug_type=problem.bug_type,
            )

        counter = Counter(all_codes)
        best_fix = counter.most_common(1)[0][0]
        result = self.executor.execute(best_fix, problem.test_code)

        # Also check if any individual sample passes
        first_success = False
        for c in all_codes:
            r = self.executor.execute(c, problem.test_code)
            if r["passed"]:
                first_success = True
                if not result["passed"]:
                    best_fix = c
                    result = r
                break

        return DebugResult(
            task_id=problem.task_id,
            method=f"cot-sc-{self.n_samples}",
            success=result["passed"],
            fix_code=best_fix if result["passed"] else None,
            nodes_explored=self.n_samples,
            backtracks=0,
            total_tokens=total_tokens,
            time_elapsed=round(time.time() - t0, 2),
            first_attempt_success=first_success,
            bug_type=problem.bug_type,
        )
