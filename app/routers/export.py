import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter(prefix="/api/export", tags=["export"])

class ExportRequest(BaseModel):
    code: str
    filename: str = "analysis_script"

@router.post("/python")
async def export_python_script(req: ExportRequest):
    """Exports raw python code as a downloadable .py file."""
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="No code content provided to export.")
    
    headers = {
        "Content-Disposition": f"attachment; filename={req.filename}.py"
    }
    return Response(content=req.code, media_type="text/x-python", headers=headers)

@router.post("/notebook")
async def export_jupyter_notebook(req: ExportRequest):
    """Converts raw python code string into a standard .ipynb Jupyter Notebook struct."""
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="No code content provided to export.")
    
    # Standard Jupyter Notebook Schema (nbformat v4)
    notebook_structure = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Autonomous AI Data Scientist - Execution Notebook\n",
                    "Auto-generated code pipeline executed via E2B Sandboxes."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": req.code.splitlines(keepends=True)
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    notebook_json = json.dumps(notebook_structure, indent=2)
    headers = {
        "Content-Disposition": f"attachment; filename={req.filename}.ipynb"
    }
    return Response(content=notebook_json, media_type="application/x-ipynb+json", headers=headers)