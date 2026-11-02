from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.schemas.profile import DatasetProfile
from app.schemas.sandbox import ExecutionResult

class AgentState(BaseModel):
    """Global state object passed across all LangGraph nodes."""
    filename: str                              # Dataset filename stored in uploads
    file_path: str                             # Local filesystem path
    question: str                              # User business query
    
    profile: Optional[DatasetProfile] = None   # Dataset profiling metadata
    generated_code: Optional[str] = None       # Active LLM-generated Python snippet
    execution_result: Optional[ExecutionResult] = None # Output from E2B sandbox
    
    error_context: str = "None"                # Stores error stack trace for retries
    retry_count: int = 0                       # Track self-healing retry iterations
    max_retries: int = 3                       # Guardrail to avoid infinite loops
    is_complete: bool = False                  # Final completion flag