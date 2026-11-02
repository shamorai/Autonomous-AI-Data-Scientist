# Autonomous AI Data Scientist Engine 🤖📊

An enterprise-grade, autonomous multi-agent engine that profiles raw datasets, executes Python analysis inside isolated E2B microVM sandboxes, trains AutoML models, and generates downloadable PDF executive reports.

## 🏗️ Architecture
- **Orchestration:** LangGraph state machine with self-healing reflection loop.
- **Backend:** FastAPI, Polars, Pydantic v2.
- **Sandbox Execution:** E2B Firecracker MicroVMs.
- **LLM Engine:** Groq (Llama-3.3-70b-versatile) via LangChain.
- **UI:** TailwindCSS & Jinja2 template dashboard.

## 🚀 Quick Start

1. **Clone & Setup:**
   ```bash
   git clone [https://github.com/devupadhyay01/Autonomous-Data-scientist.git](https://github.com/shamorai/Autonomous-Data-scientist.git)
   cd Autonomous-Data-scientist
   uv pip install -r requirements.txt


   what is the name of the pm 