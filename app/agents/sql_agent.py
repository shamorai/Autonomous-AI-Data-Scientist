from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # Or ChatOpenAI depending on your LLM provider
from app.schemas.profile import DatasetProfile
from app.schemas.agent import SQLAgentOutput
import os

class SQLAgent:
    """Agent responsible for translating business questions into executable SQL queries."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        # Initialize LLM with structured output capabilities
        self.llm = ChatGroq(
            temperature=0.1, 
            model_name=model_name, 
            api_key=os.getenv("GROQ_API_KEY")
        ).with_structured_output(SQLAgentOutput)

        # System prompt restricting SQL syntax to safe SELECT operations
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert SQL Data Analyst. Your job is to generate accurate SQL queries "
             "based on the provided dataset schema and business question.\n\n"
             "RULES:\n"
             "1. Only output standard SELECT queries (no UPDATE, DROP, or DELETE).\n"
             "2. The dataset table is named 'data_table'.\n"
             "3. Refer directly to the provided column names and types.\n"
             "4. Keep queries efficient and clean."),
            ("user", 
             "Dataset Profile:\n{dataset_profile}\n\n"
             "Business Question: {question}")
        ])

        self.chain = self.prompt | self.llm

    def generate_query(self, profile: DatasetProfile, question: str) -> SQLAgentOutput:
        """Executes prompt chain and returns structured SQL output."""
        # Convert profile schema into concise readable string for prompt
        profile_summary = f"Filename: {profile.filename}, Total Rows: {profile.row_count}\n"
        profile_summary += "Columns:\n"
        for col in profile.columns:
            profile_summary += f"- {col.name} ({col.dtype}): {col.null_percentage}% missing\n"

        return self.chain.invoke({
            "dataset_profile": profile_summary,
            "question": question
        })