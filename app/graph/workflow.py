import os
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from app.schemas.state import AgentState
from app.services.profiler import DataProfiler
from langgraph.checkpoint.memory import MemorySaver
from app.agents.python_agent import PythonCodeAgent
from app.services.sandbox import E2BSandboxService

# Global Memory Checkpointer Instance
checkpointer = MemorySaver()

# 1. Define Node Functions

def profiler_node(state: AgentState) -> Dict[str, Any]:
    """Node: Profiles the dataset if profile is not already present in state."""
    if not state.profile:
        profiler = DataProfiler(state.file_path)
        profile_data = profiler.profile()
        return {"profile": profile_data}
    return {}


def code_generator_node(state):
    agent = PythonCodeAgent()
    
    # Safe attribute/dict access for Pydantic AgentState object
    if hasattr(state, "profile"):
        profile = getattr(state, "profile", "") or getattr(state, "profile_summary", "")
        question = getattr(state, "question", "")
        error_context = getattr(state, "error_context", "None")
    elif isinstance(state, dict):
        profile = state.get("profile") or state.get("profile_summary") or ""
        question = state.get("question", "")
        error_context = state.get("error_context", "None")
    else:
        profile, question, error_context = "", "", "None"

    if isinstance(profile, dict) or not isinstance(profile, str):
        profile = str(profile)

    code_output = agent.generate_code(
        question=question,
        profile=profile,
        error_context=error_context
    )

    print("--- GENERATED CODE START ---")
    print(code_output)
    print("--- GENERATED CODE END ---")

    # Return updated state key dictionary
    return {
        "code_snippet": code_output,
        "generated_code": code_output
    }


def execution_node(state):
    # Safe attribute extraction
    if isinstance(state, dict):
        code_snippet = state.get("code_snippet") or state.get("generated_code") or ""
        dataset_path = state.get("file_path") or state.get("filename")
    else:
        code_snippet = getattr(state, "code_snippet", "") or getattr(state, "generated_code", "")
        dataset_path = getattr(state, "file_path", None) or getattr(state, "filename", None)

    # DIRECT CALL TO E2B SANDBOX SERVICE
    sandbox_service = E2BSandboxService()
    exec_result = sandbox_service.execute_code(code_snippet=code_snippet, dataset_path=dataset_path)

    # Extract stdout list directly into state
    stdout_output = exec_result.stdout if exec_result.stdout else [exec_result.error or "No output"]

    return {
        "execution_result": exec_result,
        "stdout": stdout_output,
        "is_complete": exec_result.success
    }


def refactor_node(state: AgentState) -> Dict[str, Any]:
    """Node: Prepares error state and increments retry counter for self-healing."""
    error_msg = state.execution_result.error if state.execution_result else "Unknown execution failure."
    new_retry_count = state.retry_count + 1
    
    return {
        "error_context": f"Attempt {new_retry_count} failed with error:\n{error_msg}",
        "retry_count": new_retry_count
    }


# 2. Define Conditional Router

def check_execution_status(state: AgentState) -> Literal["end", "refactor"]:
    """Conditional Edge: Determines whether execution succeeded or needs refactoring."""
    result = state.execution_result
    
    # If code ran successfully without errors, complete execution
    if result and result.success:
        return "end"
    
    # If max retries reached, stop to prevent infinite loops
    if state.retry_count >= state.max_retries:
        return "end"
        
    # Otherwise, trigger self-healing refactor loop
    return "refactor"


# 3. Build & Compile LangGraph StateGraph with Memory Checkpointer

def build_data_science_graph():
    """Assembles node graph with state transitions, conditional routing, and checkpointer memory."""
    workflow = StateGraph(AgentState)
    
    # Add nodes to graph
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("code_generator", code_generator_node)
    workflow.add_node("executor", execution_node)
    workflow.add_node("refactor", refactor_node)
    
    # Set entry point
    workflow.set_entry_point("profiler")
    
    # Define standard sequential edges
    workflow.add_edge("profiler", "code_generator")
    workflow.add_edge("code_generator", "executor")
    
    # Add conditional router after execution
    workflow.add_conditional_edges(
        "executor",
        check_execution_status,
        {
            "end": END,
            "refactor": "refactor"
        }
    )
    
    # Loop back from refactor to code generator for auto-correction
    workflow.add_edge("refactor", "code_generator")
    
    # COMPILE WITH MEMORY SAVER CHECKPOINTER
    return workflow.compile(checkpointer=checkpointer)