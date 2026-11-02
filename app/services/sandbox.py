import base64
import os
from pathlib import Path
from typing import List, Optional, Union
from dotenv import load_dotenv

from app.schemas.sandbox import ExecutionResult
from e2b_code_interpreter import Sandbox as E2BSandbox

load_dotenv()


class E2BSandboxService:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        if not self.api_key:
            raise ValueError("E2B_API_KEY not found in environment.")
        os.environ["E2B_API_KEY"] = self.api_key

    def execute_code(
        self,
        code_snippet: str,
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> ExecutionResult:
        try:
            sandbox = E2BSandbox.create()
            try:
                # 0. Pre-install ML packages at MicroVM level to bypass runtime pip overhead
                sandbox.commands.run(
                    "pip install shap matplotlib seaborn scikit-learn --quiet --no-cache-dir"
                )

                remote_dataset_name = "data.csv"

                # 1. Upload CSV to MicroVM
                if dataset_path and Path(dataset_path).exists():
                    with open(dataset_path, "rb") as f:
                        sandbox.files.write(remote_dataset_name, f.read())

                # 2. Setup Headless Matplotlib + Pandas
                setup_code = (
                    "import sys\n"
                    "import os\n"
                    "import matplotlib\n"
                    "matplotlib.use('Agg')\n"
                    "import matplotlib.pyplot as plt\n"
                    "import pandas as pd\n"
                    "import numpy as np\n"
                    f"if os.path.exists('{remote_dataset_name}'):\n"
                    f"    df = pd.read_csv('{remote_dataset_name}')\n"
                    "else:\n"
                    "    df = pd.DataFrame()\n"
                )

                full_code = (
                    f"{setup_code}\n{code_snippet}\nimport sys; sys.stdout.flush()\n"
                )
                
                # Execution timeout set to 180s for quick response
                execution = sandbox.run_code(full_code, timeout=180)

                logs_stdout: List[str] = []
                logs_stderr: List[str] = []

                # 3. Extract Stdout Cleanly (Splitting lines properly)
                if hasattr(execution, "logs") and execution.logs:
                    # Extract stdout
                    if hasattr(execution.logs, "stdout") and execution.logs.stdout:
                        for item in execution.logs.stdout:
                            raw_text = getattr(item, "text", str(item))
                            for line in raw_text.splitlines():
                                if line.strip():
                                    logs_stdout.append(line)

                    # Extract stderr
                    if hasattr(execution.logs, "stderr") and execution.logs.stderr:
                        for item in execution.logs.stderr:
                            raw_text = getattr(item, "text", str(item))
                            for line in raw_text.splitlines():
                                if line.strip():
                                    logs_stderr.append(line)

                # 4. Extract E2B Rich Artifacts
                if hasattr(execution, "results") and execution.results:
                    for res in execution.results:
                        if hasattr(res, "png") and res.png:
                            logs_stdout.append(f"IMAGE_BASE64:{res.png}")

                # 5. Handle Execution Errors
                if getattr(execution, "error", None):
                    err_obj = execution.error
                    err_name = getattr(err_obj, "name", "ExecutionError")
                    err_val = getattr(err_obj, "value", str(err_obj))
                    err_tb = getattr(err_obj, "traceback", "")
                    error_msg = f"{err_name}: {err_val}\n{err_tb}"

                    filtered_stdout = [
                        line
                        for line in logs_stdout
                        if not line.startswith(
                            ("Collecting ", "Downloading ", "Requirement already ")
                        )
                    ]

                    return ExecutionResult(
                        success=False,
                        stdout=filtered_stdout if filtered_stdout else [error_msg],
                        stderr=logs_stderr,
                        error=error_msg,
                    )

                clean_stdout = [line for line in logs_stdout]

                if not clean_stdout:
                    clean_stdout = [
                        "Execution completed cleanly inside E2B Sandbox MicroVM."
                    ]

                return ExecutionResult(
                    success=True, stdout=clean_stdout, stderr=logs_stderr, error=None
                )

            finally:
                sandbox.kill()

        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=[f"E2B Sandbox Exception: {str(e)}"],
                stderr=[str(e)],
                error=f"E2B Sandbox Exception: {str(e)}",
            )