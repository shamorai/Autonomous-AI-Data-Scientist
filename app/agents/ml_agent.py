
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.schemas.profile import DatasetProfile
from app.schemas.ml import MLTaskDetectionOutput, MLCodeGeneratorOutput

class MLRouterAgent:
    """Agent responsible for auto-detecting ML task types and generating training pipelines."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.api_key = os.getenv("GROQ_API_KEY")
        
        # LLM instance for task routing
        self.router_llm = ChatGroq(
            temperature=0.1,
            model_name=model_name,
            api_key=self.api_key
        ).with_structured_output(MLTaskDetectionOutput)

        # LLM instance for training script generation
        self.code_llm = ChatGroq(
            temperature=0.1,
            model_name=model_name,
            api_key=self.api_key
        ).with_structured_output(MLCodeGeneratorOutput)

        # Prompt for task detection
        self.detect_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert AutoML Lead. Analyze the dataset profile and identify the most logical "
             "machine learning task type.\n\n"
             "RULES:\n"
             "1. If user target column has few unique values/categorical: 'classification'.\n"
             "2. If user target column is continuous numeric: 'regression'.\n"
             "3. If no target column is specified or goal is grouping: 'clustering'.\n"),
            ("user",
             "Dataset Profile:\n{profile_summary}\n\n"
             "User Instruction: {question}")
        ])

        # Prompt for code generation with mandatory print() rules for stdout capture
        self.code_prompt = ChatPromptTemplate.from_messages([
           
            ("system",
             "You are an expert Machine Learning Engineer. Write clean, robust Python code using Scikit-Learn / XGBoost "
             "to train and benchmark MULTIPLE baseline models on the pre-loaded DataFrame `df`.\n\n"
             "RULES:\n"
             "1. Assume DataFrame `df` is already loaded in memory.\n"
             "2. Handle missing values using SimpleImputer or fillna (median for continuous, mode for categorical).\n"
             "3. One-hot encode categorical features using `pd.get_dummies(..., drop_first=True)` or `OneHotEncoder`.\n"
             "4. Split data into train/test sets (80/20 ratio).\n"
             "5. MULTI-MODEL BENCHMARKING REQUIREMENT:\n"
             "   - For Classification: Train LogisticRegression, RandomForestClassifier, and GradientBoostingClassifier.\n"
             "   - For Regression: Train Ridge/LinearRegression, RandomForestRegressor, and GradientBoostingRegressor.\n"
             "6. Create a comparison pandas DataFrame containing Model Name, Primary Metric (Accuracy/R2), and Secondary Metric (Log Loss/RMSE).\n"
             "7. MANDATORY STDOUT PRINTING:\n"
             "   - Print the benchmark comparative scorecard clearly using `print('\\n=== MODEL BENCHMARK SCORECARD ===\\n')` and `print(benchmark_df.to_string(index=False))`.\n"
             "   - Print feature importances for the best-performing model.\n"
             "8. Do not include markdown code backticks inside the `python_code` string field."),
            ("user",
             "Task Details:\nTask Type: {task_type}\nTarget: {target_column}\nFeatures: {features}\n\n"
             "Dataset Profile:\n{profile_summary}\n\n"
             "Error Context (if retrying): {error_context}")
        ])

    def detect_task(self, profile: DatasetProfile, question: str) -> MLTaskDetectionOutput:
        """Determines task type and column mapping."""
        summary = f"Columns: {', '.join([f'{c.name} ({c.dtype})' for c in profile.columns])}"
        chain = self.detect_prompt | self.router_llm
        return chain.invoke({"profile_summary": summary, "question": question})

    def generate_ml_code(
        self,
        profile: DatasetProfile,
        task_info: MLTaskDetectionOutput,
        error_context: str = "None"
    ) -> MLCodeGeneratorOutput:
        """Generates Scikit-Learn training script."""
        summary = f"Filename: {profile.filename}, Rows: {profile.row_count}\n"
        for col in profile.columns:
            summary += f"- {col.name} ({col.dtype}): missing={col.null_percentage}%\n"

        chain = self.code_prompt | self.code_llm
        return chain.invoke({
            "task_type": task_info.task_type,
            "target_column": task_info.target_column or "None",
            "features": ", ".join(task_info.feature_columns),
            "profile_summary": summary,
            "error_context": error_context
        })