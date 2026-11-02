from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class MLTaskDetectionOutput(BaseModel):
    """Schema for automated task type detection by the ML Agent."""
    task_type: str = Field(description="Detected ML task type: 'classification', 'regression', or 'clustering'")
    target_column: Optional[str] = Field(default=None, description="Name of target variable column (None for clustering)")
    feature_columns: List[str] = Field(default_factory=list, description="List of predictor feature column names")
    reasoning: str = Field(description="Explanation of why this task type and target column were selected")

class MLCodeGeneratorOutput(BaseModel):
    """Schema for executable Scikit-Learn/XGBoost training code generation."""
    reasoning: str = Field(description="Explanation of model choice, scaling, and evaluation strategy")
    python_code: str = Field(description="Executable Python/Scikit-Learn code to run inside E2B Sandbox")
    expected_metrics: List[str] = Field(default_factory=list, description="List of expected evaluation metrics (e.g. Accuracy, F1, RMSE)")