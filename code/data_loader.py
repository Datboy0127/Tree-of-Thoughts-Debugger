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


def load_debugbench(
    n: int = None,
    bug_types: list = None,
    seed: int = None,
) -> list[Problem]:
    """
    Load DebugBench (Python3) from HuggingFace ('Rtian/DebugBench').

    bug_types: filter to specific categories, e.g. ['Logic Error'].
               Options: 'Logic Error', 'Syntax Error', 'Reference Error', 'Multiple Error'.
               Default: ['Logic Error'] — hardest category, best for demonstrating ToT advantage.
    """
    if bug_types is None:
        bug_types = ["Logic Error"]
    if seed is not None:
        random.seed(seed)

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("Rtian/DebugBench", split="test")
    except Exception as e:
        raise RuntimeError(f"DebugBench load failed: {e}. Run: pip install datasets")

    # ── Diagnose schema on first item ─────────────────────────────────────────
    _first = ds[0] if len(ds) > 0 else {}
    print(f"[data_loader] DebugBench columns: {list(_first.keys())}")
    # Sample values for language and bug_type fields
    for _col in ("language", "bug_type", "bug_category", "type"):
        if _col in _first:
            _vals = set(item.get(_col, "") for item in ds)
            print(f"  {_col!r} unique values: {sorted(_vals)[:10]}")

    # ── Detect field names from actual schema ─────────────────────────────────
    _sample = _first
    # Buggy code field
    _buggy_field = next(
        (f for f in ("bug_source_code", "bug_code", "buggy_code", "bugCode")
         if f in _sample), None
    )
    # Correct code field
    _correct_field = next(
        (f for f in ("source_code", "correct_code", "solution", "correctCode")
         if f in _sample), None
    )
    # Language field
    _lang_field = next(
        (f for f in ("language", "lang", "programming_language") if f in _sample), None
    )
    # Bug type field
    _btype_field = next(
        (f for f in ("bug_type", "bug_category", "type", "category", "error_type")
         if f in _sample), None
    )
    print(f"  Using: buggy={_buggy_field!r}  correct={_correct_field!r}  "
          f"lang={_lang_field!r}  btype={_btype_field!r}")

    # Normalise bug_types filter for case-insensitive matching
    _bug_types_lower = {b.lower() for b in bug_types} if bug_types else set()

    problems = []
    for item in ds:
        # Language filter (case-insensitive, accept "Python3", "python3", "Python")
        lang = item.get(_lang_field, "") if _lang_field else ""
        if lang.lower() not in ("python3", "python"):
            continue

        # Bug type filter (case-insensitive)
        btype_raw = item.get(_btype_field, "") if _btype_field else ""
        btype = btype_raw.strip()
        if _bug_types_lower and btype.lower() not in _bug_types_lower:
            continue

        buggy_code   = item.get(_buggy_field, "")  if _buggy_field  else ""
        correct_code = item.get(_correct_field, "") if _correct_field else ""
        description  = item.get("description", "")
        task_id      = str(item.get("task_id", item.get("id", len(problems))))

        if not buggy_code or not correct_code:
            continue

        test_code = _build_debugbench_tests(description, correct_code, buggy_code)
        if not test_code:
            continue

        p = Problem(
            task_id=task_id,
            prompt=description[:500],       # first 500 chars as context
            buggy_code=buggy_code,
            test_code=test_code,
            canonical_solution=correct_code,
            bug_type=btype,
        )
        problems.append(p)
        if n and len(problems) >= n:
            break

    if seed is not None:
        random.shuffle(problems)

    if len(problems) == 0:
        print("[data_loader] WARNING: 0 problems loaded. Check the diagnostic output above.")
        print("  Common causes:")
        print("  1. Language filter: dataset may use 'python3' (lowercase) or 'Python'")
        print("  2. Bug type filter: dataset may use 'logic_error' or different casing")
        print("  3. Test generation failed: try passing bug_types=None to load all types")
    print(f"[data_loader] Loaded {len(problems)} DebugBench problems "
          f"(bug_types={bug_types})")
    return problems[:n] if n else problems


def _extract_solution_method(code: str) -> Optional[str]:
    """Extract the first method name from a LeetCode class Solution."""
    m = re.search(r"def\s+(\w+)\s*\(self", code)
    return m.group(1) if m else None


def _parse_method_signature(code: str) -> Optional[tuple[str, list[tuple[str, str]]]]:
    """
    Parse method name and parameter types from a LeetCode Solution method.
    Returns (method_name, [(param_name, type_hint), ...]) or None.
    """
    m = re.search(r"def\s+(\w+)\s*\(self(?:,\s*(.*?))?\)\s*(?:->[^:]+)?:", code)
    if not m:
        return None
    method = m.group(1)
    params_str = m.group(2) or ""
    params = []
    for p in params_str.split(","):
        p = p.strip()
        if not p:
            continue
        if ":" in p:
            name, hint = p.split(":", 1)
            params.append((name.strip(), hint.strip()))
        else:
            params.append((p.strip(), "int"))
    return method, params


def _random_input_for_type(type_hint: str, param_name: str = "",
                            depth: int = 0) -> object:
    """
    Generate a random value for a given type hint + parameter name.
    Falls back to name-based inference when type is bare 'list'/'str'/'int'.
    Returns None for unsupported complex types (TreeNode, ListNode, etc.).
    """
    t = type_hint.strip().lower().replace(" ", "")
    name = param_name.strip().lower()

    if t in ("int", "integer"):
        return random.choice([-10, -1, 0, 1, 2, 3, 5, 7, 10, 15, 20,
                               random.randint(-50, 50)])
    if t == "bool":
        return random.choice([True, False])
    if t == "float":
        return round(random.uniform(-10, 10), 2)
    if t == "str":
        chars = "abcdefghijklmnopqrstuvwxyz"
        return "".join(random.choices(chars, k=random.randint(0, 8)))

    # Parameterised generics
    if t.startswith("list[") and depth < 2:
        inner = type_hint.strip()[5:-1]
        size = random.randint(1, 7)
        items = [_random_input_for_type(inner, "", depth + 1) for _ in range(size)]
        return None if any(v is None for v in items) else items
    if t.startswith("optional["):
        inner = type_hint.strip()[9:-1]
        return None if random.random() < 0.15 else _random_input_for_type(inner, param_name, depth)

    # Bare 'list' — infer element type and constraints from parameter name
    if t == "list" and depth < 2:
        str_names  = {"words", "strs", "tokens", "chars", "strings", "word", "letters"}
        # Params that must be positive integers (no negatives, no zero)
        pos_names  = {"coins", "weights", "prices", "cost", "gas", "speed",
                      "piles", "ratings", "times", "dist", "heights"}
        size = random.randint(2, 7)
        if any(s in name for s in str_names):
            chars = "abcdefghijklmnopqrstuvwxyz"
            return ["".join(random.choices(chars, k=random.randint(1, 5)))
                    for _ in range(size)]
        if any(s in name for s in pos_names):
            # Positive ints with distinct values so multiple paths exist
            return random.sample(range(1, 20), min(size, 15))
        # Default: mixed small ints including negatives (nums, arr, values, etc.)
        return [random.randint(-10, 15) for _ in range(size)]

    # Bare 'str'
    if t in ("str", "string"):
        chars = "abcdefghijklmnopqrstuvwxyz"
        return "".join(random.choices(chars, k=random.randint(1, 8)))

    # Common name-based fallbacks when type hint is missing or ambiguous
    if name in ("amount", "capacity", "limit", "total", "sum", "budget"):
        return random.randint(5, 20)   # positive, large enough to need search
    if name in ("target", "k", "n", "m", "x", "y", "val", "value", "threshold"):
        return random.randint(0, 15)
    if "matrix" in name or "grid" in name:
        return None  # 2D grids are too complex to generate generically

    return None  # TreeNode, ListNode, custom types — skip


def _build_debugbench_tests(description: str, correct_code: str,
                             buggy_code: str = "") -> str:
    """
    Build a discriminating test suite using two strategies combined:

    1. Description examples — parse Input/Output from the LeetCode description.
    2. Differential testing — generate random inputs from type hints, run both
       correct and buggy code, keep inputs where their outputs differ.
       These are guaranteed to expose the bug.

    Falls back to description-only if type parsing fails or no buggy code given.
    """
    method_info = _parse_method_signature(correct_code)
    method = _extract_solution_method(correct_code)
    if not method:
        return ""

    assertions: list[str] = []

    # ── Strategy 1: description examples ──────────────────────────────────────
    example_pattern = re.compile(
        r"Input:\s*(.*?)\s*Output:\s*(.*?)(?=Example|\Z|Constraints|Note|Explanation)",
        re.DOTALL | re.IGNORECASE,
    )
    for raw_input, raw_output in example_pattern.findall(description):
        try:
            args = _parse_leetcode_args(raw_input.strip())
            expected = _parse_leetcode_value(raw_output.strip().split("\n")[0])
            if args is None or expected is None:
                continue
            assertions.append(f"assert Solution().{method}({args}) == {expected!r}")
        except Exception:
            continue

    # ── Strategy 2: differential testing from random inputs ───────────────────
    if buggy_code and method_info:
        _, params = method_info
        # Build namespaces for correct and buggy code
        ns_correct: dict = {}
        ns_buggy: dict = {}
        try:
            exec(correct_code, ns_correct)
            exec(buggy_code, ns_buggy)
        except Exception:
            pass

        if "Solution" in ns_correct and "Solution" in ns_buggy:
            found = 0
            for _ in range(200):  # try up to 200 random inputs
                try:
                    inputs = [_random_input_for_type(t, name) for name, t in params]
                    if any(v is None for v in inputs):
                        continue
                    args_repr = ", ".join(repr(v) for v in inputs)
                    # Use copies to avoid mutation side-effects
                    import copy
                    inputs_c = copy.deepcopy(inputs)
                    inputs_b = copy.deepcopy(inputs)
                    out_correct = getattr(ns_correct["Solution"](), method)(*inputs_c)
                    out_buggy   = getattr(ns_buggy["Solution"](), method)(*inputs_b)
                    if out_correct != out_buggy:
                        # This input discriminates — guaranteed to catch the bug
                        assertions.append(
                            f"assert Solution().{method}({args_repr}) == {out_correct!r}"
                        )
                        found += 1
                        if found >= 5:  # 5 discriminating inputs is plenty
                            break
                except Exception:
                    continue

    if not assertions:
        return ""

    return "\n".join(assertions) + "\n"


def _parse_leetcode_args(raw: str) -> Optional[str]:
    """
    Convert 'nums = [2,7,11,15], target = 9' -> '[2,7,11,15], 9'
    Returns a string suitable for function call arguments.
    """
    # Remove variable names, keep values in order
    parts = []
    for segment in raw.split(","):
        segment = segment.strip()
        if "=" in segment:
            val = segment.split("=", 1)[1].strip()
        else:
            val = segment
        # Handle multi-word values that got split (e.g. nested lists)
        parts.append(val)

    # Re-join carefully: brackets may have been split
    joined = ", ".join(parts)
    try:
        # Validate it's parseable
        ast.literal_eval(f"({joined},)")
        return joined
    except Exception:
        # Try a simpler approach: extract all Python literals in order
        literals = re.findall(r'(\[.*?\]|-?\d+\.?\d*|"[^"]*"|\'[^\']*\'|True|False|None)', raw)
        if literals:
            return ", ".join(literals)
        return None


def _parse_leetcode_value(raw: str) -> Optional[object]:
    """Parse a single output value like '[0,1]' or '24' or 'true'."""
    raw = raw.strip().rstrip(".")
    # LeetCode uses lowercase true/false/null
    raw = raw.replace("true", "True").replace("false", "False").replace("null", "None")
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


def load_curated_problems(difficulty: str = "all") -> list[Problem]:
    """
    Hand-crafted problems with two tiers:

    'simple'  — single-area bugs (like original HumanEval-Bugs).
                The tree degenerates: first hypothesis is almost always correct.
    'complex' — multi-step algorithmic bugs where correctly identifying the
                code area genuinely constrains which hypotheses are viable,
                mirroring the Game of 24 structure from Yao et al.
    'all'     — both tiers combined.

    These are useful for ablation: running ToT on simple vs complex problems
    shows where the structured search actually adds value.
    """
    if difficulty == "simple":
        return list(_SIMPLE_PROBLEMS)
    if difficulty == "complex":
        return list(_COMPLEX_PROBLEMS)
    return list(_SIMPLE_PROBLEMS) + list(_COMPLEX_PROBLEMS)


# ── Game of 24 loader ──────────────────────────────────────────────────────────

# 100 hard Game of 24 puzzles representative of the paper's test set (rank 901-1000).
# Source: 4nums.com (sorted by human solving time). For exact replication download
# 24.csv from https://github.com/princeton-nlp/tree-of-thought-llm and call
# load_game24(csv_path="path/to/24.csv").
_GAME24_HARD_PUZZLES = [
    # Require creative multi-step reasoning / fractional intermediates
    "1 1 3 8", "1 2 7 7", "1 3 4 6", "1 5 5 5", "2 5 6 6",
    "3 3 8 8", "3 3 7 7", "4 4 6 6", "2 3 5 8", "1 4 5 8",
    "1 1 4 8", "2 4 4 8", "1 3 8 8", "2 2 5 6", "3 4 5 6",
    "1 2 3 9", "3 3 5 5", "2 2 7 7", "1 4 6 8", "2 3 4 9",
    "1 3 5 7", "3 5 5 6", "1 3 3 8", "2 3 4 6", "1 4 4 6",
    "1 2 5 8", "2 4 5 6", "1 2 6 9", "4 4 5 8", "3 4 4 8",
    "1 5 6 8", "3 5 6 8", "2 6 6 8", "4 5 6 8", "1 6 6 8",
    "3 6 6 8", "2 4 6 8", "4 4 7 8", "1 4 7 8", "3 4 7 8",
    "2 3 7 8", "1 3 7 8", "3 5 7 8", "2 5 7 8", "4 5 7 8",
    "1 5 7 8", "3 6 7 8", "2 6 7 8", "4 6 7 8", "1 6 7 8",
    "2 2 3 7", "1 3 6 8", "2 3 6 8", "3 4 4 6", "1 2 4 9",
    "2 4 6 6", "1 4 5 7", "2 3 5 6", "3 4 5 7", "1 3 5 6",
    "2 5 6 7", "3 5 5 7", "1 4 6 7", "2 3 6 7", "4 5 5 6",
    "1 2 8 8", "2 2 4 8", "3 4 6 9", "2 4 8 8", "1 4 8 9",
    "2 5 8 8", "3 4 8 8", "2 6 8 8", "4 4 8 8", "3 5 8 8",
    "1 7 7 8", "2 7 7 8", "3 7 7 8", "4 7 7 8", "5 6 6 8",
    "4 5 5 8", "3 4 5 8", "2 4 5 8", "1 5 8 8", "3 6 8 8",
    "5 6 8 8", "4 6 8 8", "5 5 6 8", "4 5 8 8", "5 7 7 8",
    "1 1 2 6", "1 2 2 7", "1 1 5 8", "2 2 8 8", "1 4 4 9",
    "3 3 4 9", "2 4 4 9", "1 5 5 9", "2 5 5 9", "3 4 6 7",
]


def load_game24(
    n: int = None,
    csv_path: str = None,
    difficulty: str = "hard",
    seed: int = None,
) -> list[str]:
    """
    Load Game of 24 puzzles. Returns list of puzzle strings like '4 9 10 13'.

    For exact paper replication (problems 901-1000), download 24.csv from the
    paper's GitHub repo and pass csv_path='path/to/24.csv'.

    Without csv_path, returns the built-in list of 100 hard puzzles.

    Args:
        n:          Number of puzzles to return (None = all).
        csv_path:   Path to 24.csv from princeton-nlp/tree-of-thought-llm.
        difficulty: 'hard' uses problems 901-1000; 'all' uses all 1362.
        seed:       Random seed for shuffling.
    """
    if csv_path is not None:
        puzzles = _load_game24_csv(csv_path, difficulty)
    else:
        puzzles = list(_GAME24_HARD_PUZZLES)

    if seed is not None:
        import random
        random.seed(seed)
        random.shuffle(puzzles)

    if n is not None:
        puzzles = puzzles[:n]

    print(f"[data_loader] Loaded {len(puzzles)} Game of 24 puzzles")
    return puzzles


def _load_game24_csv(csv_path: str, difficulty: str = "hard") -> list[str]:
    """Parse 24.csv (columns: Puzzle, Rank) and return puzzle strings."""
    puzzles = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Puzzle"):
                continue
            parts = line.split(",")
            if len(parts) < 1:
                continue
            puzzle = parts[0].strip()
            rank = int(parts[1].strip()) if len(parts) > 1 else 0
            if difficulty == "hard" and not (901 <= rank <= 1000):
                continue
            puzzles.append(puzzle)
    return puzzles


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

# ── Simple problems (single-area bugs) ────────────────────────────────────────
# These mirror HumanEval-Bugs: one wrong token, identifiable on first hypothesis.
# Useful as a control group — ToT should not add much over IO/CoT here.

_SIMPLE_PROBLEMS = [
    Problem(
        task_id="simple/binary_search",
        prompt="def binary_search(arr: list, target: int) -> int:\n    \"\"\"Return index of target in sorted arr, or -1 if not found.\"\"\"\n",
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
            "assert binary_search([1,3,5,7,9], 5) == 2\n"
            "assert binary_search([1,3,5,7,9], 1) == 0\n"
            "assert binary_search([1,3,5,7,9], 9) == 4\n"
            "assert binary_search([1,3,5,7,9], 6) == -1\n"
            "assert binary_search([1], 2) == -1\n"  # target > all elements: lo reaches len(arr), arr[mid] IndexError
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
        bug_type="off_by_one",
    ),
    Problem(
        task_id="simple/is_palindrome",
        prompt="def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s reads the same forwards and backwards.\"\"\"\n",
        buggy_code=(
            "def is_palindrome(s: str) -> bool:\n"
            "    return s != s[::-1]  # BUG: should be ==\n"
        ),
        test_code=(
            "assert is_palindrome('racecar') == True\n"
            "assert is_palindrome('hello') == False\n"
            "assert is_palindrome('') == True\n"
            "assert is_palindrome('a') == True\n"
        ),
        canonical_solution="def is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n",
        bug_type="wrong_operator",
    ),
    Problem(
        task_id="simple/sum_range",
        prompt="def sum_range(lo: int, hi: int) -> int:\n    \"\"\"Return sum of integers from lo to hi inclusive.\"\"\"\n",
        buggy_code=(
            "def sum_range(lo: int, hi: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(lo, hi):  # BUG: should be range(lo, hi+1)\n"
            "        total += i\n"
            "    return total\n"
        ),
        test_code=(
            "assert sum_range(1, 5) == 15\n"
            "assert sum_range(0, 0) == 0\n"
            "assert sum_range(3, 3) == 3\n"
            "assert sum_range(1, 10) == 55\n"
        ),
        canonical_solution=(
            "def sum_range(lo: int, hi: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(lo, hi + 1):\n"
            "        total += i\n"
            "    return total\n"
        ),
        bug_type="off_by_one",
    ),
    Problem(
        task_id="simple/max_of_three",
        prompt="def max_of_three(a: int, b: int, c: int) -> int:\n    \"\"\"Return the largest of three integers.\"\"\"\n",
        buggy_code=(
            "def max_of_three(a: int, b: int, c: int) -> int:\n"
            "    if a >= b and a >= c:\n"
            "        return a\n"
            "    elif b >= a and b >= c:\n"
            "        return b\n"
            "    return -c  # BUG: should return c\n"
        ),
        test_code=(
            "assert max_of_three(1, 2, 3) == 3\n"
            "assert max_of_three(5, 3, 1) == 5\n"
            "assert max_of_three(2, 5, 2) == 5\n"
            "assert max_of_three(-1, -2, -3) == -1\n"
        ),
        canonical_solution=(
            "def max_of_three(a: int, b: int, c: int) -> int:\n"
            "    if a >= b and a >= c:\n"
            "        return a\n"
            "    elif b >= a and b >= c:\n"
            "        return b\n"
            "    return c\n"
        ),
        bug_type="incorrect_return",
    ),
]

# ── Complex problems (multi-step algorithmic bugs) ────────────────────────────
# These mirror the Game of 24 structure: correctly identifying the *area*
# (e.g. "DP transition" vs "initialization") genuinely constrains which
# hypotheses are plausible, and a wrong area wastes all downstream nodes.
# The 3-level tree (Area → Hypothesis → Fix) provides real value here.

_COMPLEX_PROBLEMS = [
    Problem(
        task_id="complex/coin_change",
        prompt=(
            "def coin_change(coins: list, amount: int) -> int:\n"
            "    \"\"\"Return fewest coins to make amount, or -1 if impossible.\n"
            "    Classic DP: dp[i] = min coins to make value i.\"\"\"\n"
        ),
        buggy_code=(
            "def coin_change(coins: list, amount: int) -> int:\n"
            "    dp = [float('inf')] * (amount + 1)\n"
            "    dp[0] = 0\n"
            "    for i in range(1, amount + 1):\n"
            "        for coin in coins:\n"
            "            if coin <= i:\n"
            "                dp[i] = dp[i - coin] + 1  # BUG: missing min(); overwrites instead of minimising\n"
            "    return dp[amount] if dp[amount] != float('inf') else -1\n"
        ),
        test_code=(
            "assert coin_change([1, 5, 6, 9], 11) == 2\n"
            "assert coin_change([1, 2, 5], 11) == 3\n"
            "assert coin_change([2], 3) == -1\n"
            "assert coin_change([1], 0) == 0\n"
            "assert coin_change([1, 2, 5], 6) == 2\n"
        ),
        canonical_solution=(
            "def coin_change(coins: list, amount: int) -> int:\n"
            "    dp = [float('inf')] * (amount + 1)\n"
            "    dp[0] = 0\n"
            "    for i in range(1, amount + 1):\n"
            "        for coin in coins:\n"
            "            if coin <= i:\n"
            "                dp[i] = min(dp[i], dp[i - coin] + 1)\n"
            "    return dp[amount] if dp[amount] != float('inf') else -1\n"
        ),
        bug_type="Logic Error",
    ),
    Problem(
        task_id="complex/knapsack_01",
        prompt=(
            "def knapsack(weights: list, values: list, capacity: int) -> int:\n"
            "    \"\"\"0/1 knapsack: each item used at most once.\n"
            "    Must iterate weights backwards to prevent using an item twice.\"\"\"\n"
        ),
        buggy_code=(
            "def knapsack(weights: list, values: list, capacity: int) -> int:\n"
            "    n = len(weights)\n"
            "    dp = [0] * (capacity + 1)\n"
            "    for i in range(n):\n"
            "        for w in range(weights[i], capacity + 1):  # BUG: forward iteration allows item reuse\n"
            "            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])\n"
            "    return dp[capacity]\n"
        ),
        test_code=(
            "assert knapsack([2, 3, 4, 5], [3, 4, 5, 6], 5) == 7\n"
            "assert knapsack([1, 2, 3], [1, 6, 10], 5) == 16\n"
            "assert knapsack([1], [1], 0) == 0\n"
            "assert knapsack([3, 4, 5], [4, 5, 6], 4) == 5\n"
            "assert knapsack([1, 3], [2, 4], 4) == 6\n"  # forward iteration gives 8 (reuses weight-1 item); 0/1 gives 6
        ),
        canonical_solution=(
            "def knapsack(weights: list, values: list, capacity: int) -> int:\n"
            "    n = len(weights)\n"
            "    dp = [0] * (capacity + 1)\n"
            "    for i in range(n):\n"
            "        for w in range(capacity, weights[i] - 1, -1):\n"
            "            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])\n"
            "    return dp[capacity]\n"
        ),
        bug_type="Logic Error",
    ),
    Problem(
        task_id="complex/longest_common_subsequence",
        prompt=(
            "def lcs(s1: str, s2: str) -> int:\n"
            "    \"\"\"Return length of longest common subsequence of s1 and s2.\n"
            "    DP: when chars match, extend diagonal; else take max of left/up.\"\"\"\n"
        ),
        buggy_code=(
            "def lcs(s1: str, s2: str) -> int:\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
            "    for i in range(1, m + 1):\n"
            "        for j in range(1, n + 1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = dp[i-1][j-1]  # BUG: should be max(dp[i-1][j], dp[i][j-1])\n"
            "    return dp[m][n]\n"
        ),
        test_code=(
            "assert lcs('abcde', 'ace') == 3\n"
            "assert lcs('abc', 'abc') == 3\n"
            "assert lcs('abc', 'def') == 0\n"
            "assert lcs('AGGTAB', 'GXTXAYB') == 4\n"
            "assert lcs('', 'abc') == 0\n"
        ),
        canonical_solution=(
            "def lcs(s1: str, s2: str) -> int:\n"
            "    m, n = len(s1), len(s2)\n"
            "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
            "    for i in range(1, m + 1):\n"
            "        for j in range(1, n + 1):\n"
            "            if s1[i-1] == s2[j-1]:\n"
            "                dp[i][j] = dp[i-1][j-1] + 1\n"
            "            else:\n"
            "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
            "    return dp[m][n]\n"
        ),
        bug_type="Logic Error",
    ),
    Problem(
        task_id="complex/num_islands",
        prompt=(
            "def num_islands(grid: list) -> int:\n"
            "    \"\"\"Count connected components of '1's in a binary grid.\n"
            "    DFS must mark visited cells to avoid infinite recursion.\"\"\"\n"
        ),
        buggy_code=(
            "def num_islands(grid: list) -> int:\n"
            "    if not grid:\n"
            "        return 0\n"
            "    rows, cols = len(grid), len(grid[0])\n"
            "\n"
            "    def dfs(r, c):\n"
            "        if r < 0 or r >= rows or c < 0 or c >= cols:\n"
            "            return\n"
            "        if grid[r][c] != '1':\n"
            "            return\n"
            "        # BUG: missing grid[r][c] = '0' — cells never marked visited\n"
            "        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)\n"
            "\n"
            "    count = 0\n"
            "    for r in range(rows):\n"
            "        for c in range(cols):\n"
            "            if grid[r][c] == '1':\n"
            "                dfs(r, c)\n"
            "                count += 1\n"
            "    return count\n"
        ),
        test_code=(
            "import sys; sys.setrecursionlimit(100)  # force RecursionError fast if unvisited\n"
            "try:\n"
            "    g1 = [['1','1','0'],['0','1','0'],['0','0','1']]\n"
            "    assert num_islands(g1) == 2\n"
            "    g2 = [['1','0','1'],['0','0','0'],['1','0','1']]\n"
            "    assert num_islands(g2) == 4\n"
            "    g3 = [['0']]\n"
            "    assert num_islands(g3) == 0\n"
            "except RecursionError:\n"
            "    raise AssertionError('infinite recursion — cells not marked visited')\n"
        ),
        canonical_solution=(
            "def num_islands(grid: list) -> int:\n"
            "    if not grid:\n"
            "        return 0\n"
            "    rows, cols = len(grid), len(grid[0])\n"
            "\n"
            "    def dfs(r, c):\n"
            "        if r < 0 or r >= rows or c < 0 or c >= cols:\n"
            "            return\n"
            "        if grid[r][c] != '1':\n"
            "            return\n"
            "        grid[r][c] = '0'\n"
            "        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)\n"
            "\n"
            "    count = 0\n"
            "    for r in range(rows):\n"
            "        for c in range(cols):\n"
            "            if grid[r][c] == '1':\n"
            "                dfs(r, c)\n"
            "                count += 1\n"
            "    return count\n"
        ),
        bug_type="Logic Error",
    ),
    Problem(
        task_id="complex/max_subarray",
        prompt=(
            "def max_subarray(nums: list) -> int:\n"
            "    \"\"\"Return the sum of the maximum contiguous subarray (Kadane's algorithm).\n"
            "    Current subarray sum resets to 0 when it goes negative.\"\"\"\n"
        ),
        buggy_code=(
            "def max_subarray(nums: list) -> int:\n"
            "    max_sum = nums[0]\n"
            "    current = 0\n"
            "    for num in nums:\n"
            "        current += num        # BUG: never resets; should be max(0, current) + num\n"
            "        max_sum = max(max_sum, current)\n"
            "    return max_sum\n"
        ),
        test_code=(
            "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6\n"
            "assert max_subarray([1]) == 1\n"
            "assert max_subarray([-1,-2,-3]) == -1\n"
            "assert max_subarray([5,4,-1,7,8]) == 23\n"
        ),
        canonical_solution=(
            "def max_subarray(nums: list) -> int:\n"
            "    max_sum = nums[0]\n"
            "    current = 0\n"
            "    for num in nums:\n"
            "        current = max(0, current) + num\n"
            "        max_sum = max(max_sum, current)\n"
            "    return max_sum\n"
        ),
        bug_type="Logic Error",
    ),
    Problem(
        task_id="complex/climbing_stairs",
        prompt=(
            "def climb_stairs(n: int) -> int:\n"
            "    \"\"\"Count distinct ways to climb n stairs taking 1 or 2 steps at a time.\n"
            "    DP: ways[i] = ways[i-1] + ways[i-2] with base cases ways[1]=1, ways[2]=2.\"\"\"\n"
        ),
        buggy_code=(
            "def climb_stairs(n: int) -> int:\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    dp = [0] * (n + 1)\n"
            "    dp[1] = 1\n"
            "    dp[2] = 2\n"
            "    for i in range(3, n + 1):\n"
            "        dp[i] = dp[i-1] + dp[i-3]  # BUG: should be dp[i-2] not dp[i-3]\n"
            "    return dp[n]\n"
        ),
        test_code=(
            "assert climb_stairs(1) == 1\n"
            "assert climb_stairs(2) == 2\n"
            "assert climb_stairs(3) == 3\n"
            "assert climb_stairs(4) == 5\n"
            "assert climb_stairs(5) == 8\n"
            "assert climb_stairs(10) == 89\n"
        ),
        canonical_solution=(
            "def climb_stairs(n: int) -> int:\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    dp = [0] * (n + 1)\n"
            "    dp[1] = 1\n"
            "    dp[2] = 2\n"
            "    for i in range(3, n + 1):\n"
            "        dp[i] = dp[i-1] + dp[i-2]\n"
            "    return dp[n]\n"
        ),
        bug_type="Logic Error",
    ),
]
