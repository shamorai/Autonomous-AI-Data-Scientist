

from pydantic import BaseModel
from typing import Optional, List, Union, Any

class ExecutionResult(BaseModel):
    success: bool
    stdout: Union[List[str], str] = []
    stderr: Union[List[str], str] = []
    error: Optional[str] = None
    charts: Optional[List[Any]] = []
    results: Optional[List[Any]] = []