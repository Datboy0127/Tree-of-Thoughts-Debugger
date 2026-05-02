"""Safe sandboxed Python code execution with timeout."""
import subprocess
import tempfile
import os
import re
import textwrap
import config


class CodeExecutor:
    def __init__(self, timeout: int = None):
        self.timeout = timeout or config.EXEC_TIMEOUT

    def execute(self, code: str, test_code: str) -> dict:
        """Execute `code` + `test_code` in a subprocess. Returns result dict."""
        full_source = self._build_source(code, test_code)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_source)
            tmpfile = f.name

        try:
            proc = subprocess.run(
                ["python3", tmpfile],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "passed": proc.returncode == 0,
                "stdout": proc.stdout[:2000],
                "stderr": proc.stderr[:2000],
                "returncode": proc.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout}s",
                "returncode": -1,
                "timed_out": True,
            }
        except Exception as e:
            return {
                "passed": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "timed_out": False,
            }
        finally:
            try:
                os.unlink(tmpfile)
            except OSError:
                pass

    def count_passing_tests(self, code: str, test_cases: list[dict]) -> tuple[int, int]:
        """Run individual test cases. Returns (passed, total)."""
        passed = 0
        for tc in test_cases:
            test_src = f"assert {tc['input']} == {tc['expected']}, 'failed'"
            result = self.execute(code, test_src)
            if result["passed"]:
                passed += 1
        return passed, len(test_cases)

    @staticmethod
    def _build_source(code: str, test_code: str) -> str:
        """Combine function code and test harness."""
        code = textwrap.dedent(code).strip()
        test_code = textwrap.dedent(test_code).strip()
        return f"{code}\n\n{test_code}\n"

    @staticmethod
    def extract_function_name(code: str) -> str:
        match = re.search(r"def\s+(\w+)\s*\(", code)
        return match.group(1) if match else ""
