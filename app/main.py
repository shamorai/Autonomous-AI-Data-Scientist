import os
import re
import traceback
import joblib
from pathlib import Path
from typing import Optional, Any
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from app.services.profiler import DataProfiler
from app.schemas.profile import DatasetProfile
from app.services.sandbox import E2BSandboxService
from app.schemas.sandbox import ExecutionResult
from app.schemas.agent import SQLAgentOutput, PythonAgentOutput
from app.agents.sql_agent import SQLAgent
from app.agents.python_agent import PythonCodeAgent
from app.graph.workflow import build_data_science_graph
from app.schemas.state import AgentState
from app.schemas.ml import MLTaskDetectionOutput, MLCodeGeneratorOutput
from app.agents.ml_agent import MLRouterAgent
from app.routers import export

# Initialize FastAPI application
app = FastAPI(title="Autonomous AI Data Scientist - Backend API")

# Mount export routes (e.g. /api/export/python, /api/export/notebook)
app.include_router(export.router)

# Define directories
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Templates directory for Web UI
templates = Jinja2Templates(directory="app/templates")

# Compile LangGraph graph workflow once at module startup
compiled_graph: Any = build_data_science_graph()



@app.get("/")
async def serve_dashboard(request: Request):
    """Serves the main interactive web dashboard."""
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "Autonomous AI Data Scientist Engine"}


@app.post("/api/profile", response_model=DatasetProfile)
async def profile_dataset(file: UploadFile = File(...)):
    """Receives uploaded CSV file, saves it locally, runs profiler engine, and returns JSON profile."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are currently supported.")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        profiler = DataProfiler(file_path)
        profile_data = profiler.profile()
        return profile_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")


@app.post("/api/execute", response_model=ExecutionResult)
async def execute_code_in_sandbox(
    filename: str = Form(...),
    code: str = Form(...)
):
    """Executes Python code directly inside E2B Cloud MicroVM."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file '{filename}' not found in uploads.")

    try:
        sandbox_service = E2BSandboxService()
        result = sandbox_service.execute_code(code_snippet=code, dataset_path=file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sandbox execution error: {str(e)}")


@app.post("/api/agent/python", response_model=PythonAgentOutput)
async def generate_python_analysis(
    filename: str = Form(...),
    question: str = Form(...)
):
    """Generates analytical Pandas code based on an uploaded dataset profile."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found.")
        
    profiler = DataProfiler(file_path)
    profile = profiler.profile()
    
    agent = PythonCodeAgent()
    result = agent.generate_code(profile=profile, question=question)
    return result


@app.post("/api/workflow/run")
async def run_analysis_workflow(
    filename: str = Form(...),
    question: str = Form(...)
):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file '{filename}' not found.")

    input_data = {
        "filename": filename,
        "file_path": str(file_path),
        "question": question,
        "error_context": "None",
        "retry_count": 0,
        "max_retries": 3,
        "is_complete": False
    }

    config = {
        "configurable": {
            "thread_id": filename
        }
    }

    try:
        final_state = compiled_graph.invoke(input_data, config=config)
        
        if isinstance(final_state, dict):
            code = final_state.get("code_snippet") or final_state.get("generated_code") or ""
            stdout_list = final_state.get("stdout") or []
            exec_res = final_state.get("execution_result") or {}
        else:
            code = getattr(final_state, "code_snippet", "") or getattr(final_state, "generated_code", "")
            stdout_list = getattr(final_state, "stdout", [])
            exec_res = getattr(final_state, "execution_result", {})

        if not stdout_list:
            if hasattr(exec_res, "stdout") and exec_res.stdout:
                stdout_list = exec_res.stdout
            elif isinstance(exec_res, dict) and exec_res.get("stdout"):
                stdout_list = exec_res.get("stdout")

        if not stdout_list or stdout_list == []:
            stdout_list = ["Code executed cleanly in E2B Sandbox container."]

        return {
            "status": "success",
            "code": code,
            "stdout": stdout_list,
            "stderr": getattr(exec_res, "stderr", []) if hasattr(exec_res, "stderr") else exec_res.get("stderr", []),
            "error": getattr(exec_res, "error", None) if hasattr(exec_res, "error") else exec_res.get("error", None)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/train")
async def train_automl_model(
    filename: str = Form(...),
    question: str = Form(...)
):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file '{filename}' not found.")

    input_data = {
        "filename": filename,
        "file_path": str(file_path),
        "question": question,
        "error_context": "None",
        "retry_count": 0,
        "max_retries": 3,
        "is_complete": False
    }

    config = {
        "configurable": {
            "thread_id": filename
        }
    }

    try:
        final_state = compiled_graph.invoke(input_data, config=config)

        if isinstance(final_state, dict):
            code = final_state.get("code_snippet") or final_state.get("generated_code") or ""
            exec_res = final_state.get("execution_result") or {}
            stdout_list = final_state.get("stdout") or []
        else:
            code = getattr(final_state, "code_snippet", "") or getattr(final_state, "generated_code", "")
            exec_res = getattr(final_state, "execution_result", {})
            stdout_list = getattr(final_state, "stdout", [])

        if not stdout_list:
            if hasattr(exec_res, "stdout") and exec_res.stdout:
                stdout_list = exec_res.stdout
            elif isinstance(exec_res, dict) and exec_res.get("stdout"):
                stdout_list = exec_res.get("stdout", [])

        if not stdout_list or stdout_list == ["AutoML Model training completed inside E2B Sandbox MicroVM."]:
            sandbox_service = E2BSandboxService()
            direct_exec = sandbox_service.execute_code(code_snippet=code, dataset_path=file_path)
            if direct_exec.stdout and len(direct_exec.stdout) > 0:
                stdout_list = direct_exec.stdout

        return {
            "status": "success",
            "python_code": code,
            "code": code,
            "stdout": stdout_list,
            "stderr": getattr(exec_res, "stderr", []) if hasattr(exec_res, "stderr") else exec_res.get("stderr", []),
            "error": getattr(exec_res, "error", None) if hasattr(exec_res, "error") else exec_res.get("error", None)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/model")
async def export_model(filename: str = Form(...)):
    """Returns the latest trained model artifact (.joblib) for download."""
    model_path = MODEL_DIR / f"{Path(filename).stem}_model.joblib"
    
    if not model_path.exists():
        default_path = MODEL_DIR / "model.joblib"
        if default_path.exists():
            model_path = default_path
        else:
            raise HTTPException(status_code=404, detail="No trained model artifact found. Please train a model first.")

    return FileResponse(
        path=model_path,
        filename=f"{Path(filename).stem}_model.joblib",
        media_type="application/octet-stream"
    )



@app.post("/api/report/generate")
async def generate_pdf_report(
    filename: str = Form(...),
    question: str = Form(""),
    generated_code: str = Form(""),
    execution_output: str = Form(""),
    profile: str = Form(""),
):
    try:
        from app.services.report_gen import PDFReportGenerator

        output_dir = Path("app/exports")
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_filename = f"Report_{filename.replace('.csv', '')}.pdf"
        output_pdf_path = str(output_dir / pdf_filename)

        # Base64 images regex extract
        images = re.findall(
            r"IMAGE_BASE64:\s*([A-Za-z0-9+/=]+)", execution_output
        )

        context = {
            "filename": filename,
            "question": question,
            "generated_code": generated_code,
            "execution_output": execution_output,
            "profile": profile if profile else "Dataset profiling completed.",
            "images": images,
        }

        report_service = PDFReportGenerator()
        generated_file_path = report_service.generate_pdf(
            context, output_pdf_path
        )

        if os.path.exists(generated_file_path):
            media_type = (
                "application/pdf"
                if generated_file_path.endswith(".pdf")
                else "text/html"
            )
            return FileResponse(
                path=generated_file_path,
                media_type=media_type,
                filename=os.path.basename(generated_file_path),
            )
        else:
            raise HTTPException(
                status_code=500, detail="Report file creation failed."
            )

    except Exception as e:
        print(f"Error in /api/report/generate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))