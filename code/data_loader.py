"""
Data loading utilities for HumanEval-Bugs and DebugBench.

HumanEval-Bugs: derived from OpenAI HumanEval with systematic mutations.
  HuggingFace: 'evalplus/humanevalplus' (original); bugs variant introduced manually.
  GitHub: https://github.com/evalplus/evalplus

DebugBench: 4,253 buggy instances from LeetCode (Python/Java/C++).
  HuggingFace: 'Rtian/DebugBench'
"""
import json
import os
import re
import random
import ast
from dataclasses import dataclass, field
from typing import Optional
import config


@dataclass
class Problem:
    task_id: str
    prompt: str               # function signature + docstring
    buggy_code: str           # full function with bug
    test_code: str            # assertions / test harness
    canonical_solution: str = ""
    entry_point: str = ""
    bug_type: str = "unknown"


# ── Synthetic bug introducers ──────────────────────────────────────────────────

_BUG_MUTATORS = {}


def _mutator(name):
    def decorator(fn):
        _BUG_MUTATORS[name] = fn
        return fn
    return decorator


@_mutator("off_by_one")
def _off_by_one(code: str) -> str:
    """Replace `range(n)` with `range(n-1)` or `range(n+1)`."""
    def replace(m):
        n = m.group(1)
        return f"range({n}-1)" if random.random() < 0.5 else f"range({n}+1)"
    mutated = re.sub(r"range\((\w+)\)", replace, code, count=1)
    return mutated if mutated != code else code.replace("<=", "<", 1)


@_mutator("wrong_operator")
def _wrong_operator(code: str) -> str:
    replacements = [("<=", "<"), (">=", ">"), ("==", "!="), ("+", "-"), ("*", "/")]
    random.shuffle(replacements)
    for src, dst in replacements:
        if src in code:
            return code.replace(src, dst, 1)
    return code


@_mutator("missing_condition")
def _missing_condition(code: str) -> str:
    """Remove an `if` guard."""
    lines = code.split("\n")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("if ") and i + 1 < len(lines):
            indent = len(line) - len(stripped)
            next_line = lines[i + 1]
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent > indent:
                # Remove the if, de-indent body by one level
                new_lines = lines[:i]
                for j in range(i + 1, len(lines)):
                    l = lines[j]
                    li = len(l) - len(l.lstrip())
                    if li > indent:
                        new_lines.append(" " * indent + l.lstrip())
                    else:
                        new_lines.extend(lines[j:])
                        break
                return "\n".join(new_lines)
    return code


@_mutator("incorrect_return")
def _incorrect_return(code: str) -> str:
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if "return " in line and line.strip() != "return":
            expr = line.split("return ", 1)[1].strip()
            if expr and not expr.startswith("("):
                new_expr = f"not ({expr})" if "True" in expr or "False" in expr else f"-({expr})"
                lines[i] = line.replace(f"return {expr}", f"return {new_expr}", 1)
                return "\n".join(lines)
    return code


def introduce_bug(code: str, bug_type: Optional[str] = None) -> tuple[str, str]:
    """Introduce a bug into correct code. Returns (buggy_code, bug_type)."""
    if bug_type is None:
        bug_type = random.choice(list(_BUG_MUTATORS.keys()))
    mutator = _BUG_MUTATORS.get(bug_type, _off_by_one)
    buggy = mutator(code)
    if buggy == code:
        bug_type = "wrong_operator"
        buggy = _wrong_operator(code)
    return buggy, bug_type


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_humaneval_bugs(
    n: int = None,
    use_synthetic: bool = True,
    seed: int = None,
) -> list[Problem]:
    """
    Load HumanEval-Bugs problems.

    Tries HuggingFace first; falls back to synthetic bugs from HumanEval.
    Set use_synthetic=True to always generate synthetic bugs (no API needed).
    """
    if seed is not None:
        random.seed(seed)

    if not use_synthetic:
        try:
            return _load_from_huggingface(n)
        except Exception as e:
            print(f"[data_loader] HuggingFace load failed ({e}), using synthetic bugs.")

    return _load_synthetic_humaneval(n, seed=seed)


def _load_from_huggingface(n: Optional[int]) -> list[Problem]:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []
    for item in ds:
        sol = item.get("canonical_solution", "")
        if not sol.strip():
            continue
        full_fn = item["prompt"] + sol
        buggy, bug_type = introduce_bug(full_fn)
        p = Problem(
            task_id=item["task_id"],
            prompt=item["prompt"],
            buggy_code=buggy,
            test_code=item["test"] + f"\ncheck({item['entry_point']})",
            canonical_solution=full_fn,
            entry_point=item["entry_point"],
            bug_type=bug_type,
        )
        problems.append(p)
        if n and len(problems) >= n:
            break
    return problems


def _load_synthetic_humaneval(n: Optional[int], seed: int = None) -> list[Problem]:
    """
    Small curated set of HumanEval-style problems with hand-crafted bugs
    for offline demo / CI use (no API key needed).
    """
    raw = _BUILTIN_PROBLEMS
    if seed is not None:
        random.seed(seed)
    random.shuffle(raw)
    if n:
        raw = raw[:n]
    return raw


def load_debugbench(n: int = None, lang: str = "python") -> list[Problem]:
    """Load DebugBench from HuggingFace ('Rtian/DebugBench')."""
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("Rtian/DebugBench", split="test")
        problems = []
        for item in ds:
            if item.get("language", "").lower() != lang:
                continue
            p = Problem(
                task_id=item.get("id", str(len(problems))),
                prompt=item.get("description", ""),
                buggy_code=item.get("buggy_solution", ""),
                test_code=item.get("test", ""),
                canonical_solution=item.get("canonical_solution", ""),
                bug_type=item.get("bug_type", "unknown"),
            )
            problems.append(p)
            if n and len(problems) >= n:
                break
        return problems
    except Exception as e:
        raise RuntimeError(f"DebugBench load failed: {e}. See data/README.md for setup.")


def save_problems(problems: list[Problem], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([p.__dict__ for p in problems], f, indent=2)


def load_problems(path: str) -> list[Problem]:
    with open(path) as f:
        return [Problem(**d) for d in json.load(f)]


# ── Built-in mini benchmark (no external deps) ─────────────────────────────────

_BUILTIN_PROBLEMS = [
    Problem(
        task_id="builtin/0",
        prompt="def has_close_elements(numbers: list, threshold: float) -> bool:\n    \"\"\"Check if any two elements are closer than threshold.\"\"\"\n",
        buggy_code=(
            "def has_close_elements(numbers: list, threshold: float) -> bool:\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i, len(numbers)):  # BUG: should be i+1\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
        ),
        test_code=(
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0], 0.5) == True\n"
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n"
            "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n"
        ),
        canonical_solution=(
            "def has_close_elements(numbers: list, threshold: float) -> bool:\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
        ),
        entry_point="has_close_elements",
        bug_type="off_by_one",
    ),
    Problem(
        task_id="builtin/1",
        prompt="def sum_to_n(n: int) -> int:\n    \"\"\"Return sum of integers 1 through n.\"\"\"\n",
        buggy_code=(
            "def sum_to_n(n: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(1, n):  # BUG: should be range(1, n+1)\n"
            "        total += i\n"
            "    return total\n"
        ),
        test_code=(
            "assert sum_to_n(5) == 15\n"
            "assert sum_to_n(10) == 55\n"
            "assert sum_to_n(1) == 1\n"
        ),
        canonical_solution=(
            "def sum_to_n(n: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i\n"
            "    return total\n"
        ),
        entry_point="sum_to_n",
        bug_type="off_by_one",
    ),
    Problem(
        task_id="builtin/2",
        prompt="def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s is a palindrome.\"\"\"\n",
        buggy_code=(
            "def is_palindrome(s: str) -> bool:\n"
            "    return s != s[::-1]  # BUG: should be ==\n"
        ),
        test_code=(
            "assert is_palindrome('racecar') == True\n"
            "assert is_palindrome('hello') == False\n"
            "assert is_palindrome('a') == True\n"
            "assert is_palindrome('') == True\n"
        ),
        canonical_solution=(
            "def is_palindrome(s: str) -> bool:\n"
            "    return s == s[::-1]\n"
        ),
        entry_point="is_palindrome",
        bug_type="wrong_operator",
    ),
    Problem(
        task_id="builtin/3",
        prompt="def max_element(lst: list) -> int:\n    \"\"\"Return the maximum element in lst.\"\"\"\n",
        buggy_code=(
            "def max_element(lst: list) -> int:\n"
            "    m = lst[0]\n"
            "    for i in range(len(lst)):\n"
            "        if lst[i] > m:  # correct comparison\n"
            "            m = lst[i]\n"
            "    return -m  # BUG: should return m\n"
        ),
        test_code=(
            "assert max_element([1, 2, 3]) == 3\n"
            "assert max_element([5, 3, 1, 2]) == 5\n"
            "assert max_element([-1, -2, -3]) == -1\n"
        ),
        canonical_solution=(
            "def max_element(lst: list) -> int:\n"
            "    m = lst[0]\n"
            "    for i in range(len(lst)):\n"
            "        if lst[i] > m:\n"
            "            m = lst[i]\n"
            "    return m\n"
        ),
        entry_point="max_element",
        bug_type="incorrect_return",
    ),
    Problem(
        task_id="builtin/4",
        prompt="def count_vowels(s: str) -> int:\n    \"\"\"Count number of vowels in s.\"\"\"\n",
        buggy_code=(
            "def count_vowels(s: str) -> int:\n"
            "    count = 0\n"
            "    for ch in s:\n"
            "        if ch in 'aeiou':  # BUG: misses uppercase\n"
            "            count += 1\n"
            "    return count\n"
        ),
        test_code=(
            "assert count_vowels('hello') == 2\n"
            "assert count_vowels('HELLO') == 2\n"
            "assert count_vowels('aeiouAEIOU') == 10\n"
            "assert count_vowels('xyz') == 0\n"
        ),
        canonical_solution=(
            "def count_vowels(s: str) -> int:\n"
            "    count = 0\n"
            "    for ch in s.lower():\n"
            "        if ch in 'aeiou':\n"
            "            count += 1\n"
            "    return count\n"
        ),
        entry_point="count_vowels",
        bug_type="missing_condition",
    ),
    Problem(
        task_id="builtin/5",
        prompt="def fizzbuzz(n: int) -> list:\n    \"\"\"Return FizzBuzz list from 1 to n.\"\"\"\n",
        buggy_code=(
            "def fizzbuzz(n: int) -> list:\n"
            "    result = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 3 == 0 and i % 5 == 0:\n"
            "            result.append('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            result.append('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            result.append('Fizz')  # BUG: should be 'Buzz'\n"
            "        else:\n"
            "            result.append(str(i))\n"
            "    return result\n"
        ),
        test_code=(
            "res = fizzbuzz(15)\n"
            "assert res[14] == 'FizzBuzz'\n"
            "assert res[4] == 'Buzz'\n"
            "assert res[2] == 'Fizz'\n"
            "assert res[0] == '1'\n"
        ),
        canonical_solution=(
            "def fizzbuzz(n: int) -> list:\n"
            "    result = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 3 == 0 and i % 5 == 0:\n"
            "            result.append('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            result.append('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            result.append('Buzz')\n"
            "        else:\n"
            "            result.append(str(i))\n"
            "    return result\n"
        ),
        entry_point="fizzbuzz",
        bug_type="wrong_operator",
    ),
    Problem(
        task_id="builtin/6",
        prompt="def binary_search(arr: list, target: int) -> int:\n    \"\"\"Return index of target in sorted arr, or -1.\"\"\"\n",
        buggy_code=(
            "def binary_search(arr: list, target: int) -> int:\n"
            "    lo, hi = 0, len(arr)  # BUG: should be len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        test_code=(
            "assert binary_search([1, 3, 5, 7, 9], 5) == 2\n"
            "assert binary_search([1, 3, 5, 7, 9], 1) == 0\n"
            "assert binary_search([1, 3, 5, 7, 9], 9) == 4\n"
            "assert binary_search([1, 3, 5, 7, 9], 6) == -1\n"
        ),
        canonical_solution=(
            "def binary_search(arr: list, target: int) -> int:\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        entry_point="binary_search",
        bug_type="off_by_one",
    ),
    Problem(
        task_id="builtin/7",
        prompt="def flatten(lst: list) -> list:\n    \"\"\"Flatten a nested list one level deep.\"\"\"\n",
        buggy_code=(
            "def flatten(lst: list) -> list:\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.append(item)  # BUG: should be result.extend(item)\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        test_code=(
            "assert flatten([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]\n"
            "assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]\n"
            "assert flatten([]) == []\n"
        ),
        canonical_solution=(
            "def flatten(lst: list) -> list:\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(item)\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        entry_point="flatten",
        bug_type="wrong_operator",
    ),
]
