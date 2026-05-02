# Data

## Built-in Demo Problems

Eight hand-crafted HumanEval-style problems with known bugs are bundled directly in
`code/data_loader.py` (`_BUILTIN_PROBLEMS`). No download required — these are used
automatically when you run with `--demo`.

---

## HumanEval-Bugs (recommended)

Derived from OpenAI's HumanEval benchmark. Correct solutions are mutated to introduce
realistic bugs (off-by-one, wrong operator, missing condition, incorrect return).

**Option A – Load via HuggingFace (automatic):**
```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('openai/openai_humaneval')"
```
Our `data_loader.py` automatically downloads and introduces synthetic mutations.

**Option B – Manual clone:**
```bash
git clone https://github.com/openai/human-eval.git
```
Then set `DATA_DIR` in `config.py` to point to the cloned directory.

---

## DebugBench

4,253 buggy code instances from LeetCode across Python, Java, and C++ covering 18
bug categories.

**Load via HuggingFace:**
```bash
python -c "from datasets import load_dataset; ds = load_dataset('Rtian/DebugBench')"
```
Or use `load_debugbench()` from `data_loader.py`:
```python
from data_loader import load_debugbench
problems = load_debugbench(n=100, lang="python")
```

**Reference:** Tian et al., "DebugBench: Evaluating Debugging Capability of Large Language
Models", ACL 2024. https://arxiv.org/abs/2401.04621

---

## Dataset Format

Each `Problem` object has:
| Field | Description |
|---|---|
| `task_id` | Unique identifier |
| `prompt` | Function signature + docstring |
| `buggy_code` | Full function with the introduced bug |
| `test_code` | Assertions / test harness |
| `canonical_solution` | Ground-truth correct implementation |
| `entry_point` | Function name |
| `bug_type` | `off_by_one`, `wrong_operator`, `missing_condition`, `incorrect_return` |
