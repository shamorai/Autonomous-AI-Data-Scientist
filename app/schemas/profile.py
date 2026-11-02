from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    is_numeric: bool
    sample_values: List[Any]
    stats: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Min, Max, Mean, Median, StdDev for numeric columns"
    )

class DatasetProfile(BaseModel):
    filename: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    correlations: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Pairwise Pearson correlation matrix for numeric columns"
    )
    summary_notes: List[str] = Field(
        default_factory=list,
        description="Automatic warnings (e.g., high missing values, high cardinality)"
    )