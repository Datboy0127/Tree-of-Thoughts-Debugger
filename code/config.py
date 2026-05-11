"""Configuration for Tree of Thoughts Code Debugger."""
import os

# LLM settings
# Backend: "ollama" (local, free) | "openai" | "together" | "groq"
BACKEND = os.getenv("TOT_BACKEND", "ollama")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# General-purpose model (not code-specialized) so baselines are harder
# and ToT's structured search has more room to add value — matching
# the paper's use of GPT-4 on a task it can't solve trivially.
MODEL = os.getenv("TOT_MODEL", "qwen2.5:7b")

MAX_TOKENS_AREA = 600        # level-1 area identification
MAX_TOKENS_HYPOTHESIS = 800  # level-2 hypothesis generation
MAX_TOKENS_FIX = 800         # level-3 fix generation
MAX_TOKENS_EVAL = 10         # sure / likely / impossible
TEMPERATURE = 0.7

# ToT search parameters — matched to the paper (Yao et al., NeurIPS 2023)
# Paper uses b=5 branching factor and 3-step trees for Game of 24.
TOT_K = 5           # branching factor at each level
TOT_DEPTH = 3       # depth: 1=area, 2=hypothesis, 3=fix+execution
TOT_SEARCH = "bfs"

# Evaluation mode: "llm" uses sure/likely/impossible like the paper.
# "hybrid" and "execution" are available but deviate from the paper.
EVALUATOR = "llm"

# Execution
EXEC_TIMEOUT = 5    # seconds per test execution
MAX_RETRIES = 2     # LLM call retries on failure

# Experiment
NUM_PROBLEMS = 50
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Baselines
COT_SC_SAMPLES = 5

# Random seed
SEED = 42
