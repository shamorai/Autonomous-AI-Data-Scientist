# 🤖 Autonomous AI Data Scientist Platform

> **An end-to-end, multi-tool Agentic AI Platform that autonomously executes Python/Data Science pipelines inside an isolated E2B MicroVM Sandbox, generates diagnostic ML models, and produces production-grade Executive HTML Reports.**

---

## 📋 Overview

The **Autonomous AI Data Scientist** is a full-stack, agentic platform engineered to automate raw data analysis, machine learning model training, and executive reporting without human intervention.

Given a raw CSV dataset and an analytical query, the system dynamically plans the execution, writes clean Python code, runs it inside an isolated **E2B MicroVM Sandbox**, evaluates model metrics (Confusion Matrix, SHAP plots), and exports modular artifacts (`.py` scripts, `.ipynb` notebooks, and interactive HTML executive reports).

---

## 🎯 Key Features

### 🧠 Autonomous Code Execution & Agentic Workflow

* **E2B MicroVM Sandbox Integration**: Runs untrusted dynamic Python code in an isolated containerized environment for high-security code execution.
* **Dynamic Query-to-Code Pipeline**: Translates complex NLP queries into clean Data Science pipelines (Pandas, Scikit-learn, XGBoost, LightGBM).
* **Automated Artifact Export**: Automatically outputs reproducible `.py` scripts, Jupyter Notebooks (`.ipynb`), and serialized models (`.joblib`).

### 📊 Advanced Diagnostics & Explainability (XAI)

* **Automated ML Pipelines**: Preprocessing, categorical encoding, feature scaling, and AutoML classification/regression.
* **Model Diagnostics**: Computes Confusion Matrices, Classification Reports, and performance metrics.
* **Explainable AI (SHAP)**: Generates base64-encoded SHAP feature importance charts to explain model predictions visually.
* **Executive HTML Reporting**: Generates sleek, standalone executive summary reports with embedded visualizations and execution logs.

### ⚡ Enterprise Production Architecture

* **FastAPI Backend**: Async, high-throughput REST API endpoints.
* **Dockerized Microservice**: Containerized setup ready for single-command orchestration via `docker-compose`.
* **Render Deployment**: Cloud-hosted backend infrastructure ensuring seamless availability.

---

## 🏗️ Architecture & Execution Flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          User Query / Raw Data                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Service & Controller                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LangChain / Agentic Planner                        │
│   • Generates Data Pipelines (Preprocessing + Feature Engineering)      │
│   • Builds ML Models (Random Forest / XGBoost) + Diagnostics (SHAP)     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      E2B MicroVM Code Execution                         │
│   • Isolated execution of generated Python code                         │
│   • Captures stdOut, base64 plot strings, and raw metrics               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Artifact Generator                            │
│   ┌────────────────────┬─────────────────────┬──────────────────────┐   │
│   │  Executable .py    │  Jupyter .ipynb     │  Executive HTML      │   │
│   │  Script            │  Notebook           │  Report              │   │
│   └────────────────────┴─────────────────────┴──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

```

---

## 📦 Project Structure

Autonomous-AI-Data-Scientist/
├── app/                               # Core Application Package
│   ├── agents/                        # Specialized Autonomous Agents
│   │   ├── ml_agent.py                # AutoML, diagnostics & SHAP generation agent
│   │   ├── python_agent.py            # Sandbox code synthesis & validation agent
│   │   └── sql_agent.py               # Data querying & SQL translation agent
│   │
│   ├── graph/                         # LangGraph Workflow & State Management
│   │   └── workflow.py                # State graph orchestration & node routing
│   │
│   ├── services/                      # Business & Execution Engine Services
│   │   ├── profiler.py                # Automated EDA & dataset profiling service
│   │   ├── report_gen.py              # Dynamic HTML/Jinja2 report compiler
│   │   └── sandbox.py                 # Isolated E2B MicroVM execution manager
│   │
│   ├── routers/                       # FastAPI Route Controllers
│   │   └── export.py                  # API endpoints for pipeline & report exports
│   │
│   ├── schemas/                       # Pydantic Schemas & Data Validation
│   │   ├── agent.py                   # Agent payload & response schemas
│   │   ├── ml.py                      # Model evaluation & metrics data models
│   │   ├── profile.py                 # Profiling output schemas
│   │   ├── sandbox.py                 # Sandbox execution state models
│   │   └── state.py                   # LangGraph global state schema
│   │
│   ├── templates/                     # Executive Report Templates
│   │   ├── index.html                 # Dashboard base template
│   │   └── report.html                # Executive summary report layout
│   │
│   └── exports/                       # Compiled Artifact Outputs
│       └── Report_employee_records.html
│
├── data/                              # Data Persistence Layer
│   ├── uploads/                       # Raw input CSV datasets
│   ├── models/                        # Serialized ML model artifacts (.joblib)
│   └── reports/                       # Generated HTML/PDF executive reports
│
├── test/                              # Suite for Unit & Integration Tests
│   └── test_workflow.py               # Automated tests for LangGraph execution & node routing
│
├── main.py                            # FastAPI Application Entry Point
├── Dockerfile                         # Production Container Spec
├── docker-compose.yml                 # Multi-container Microservice Setup
├── requirements.txt                   # Project Dependencies
├── LICENSE                            # MIT License
└── README.md                          # Project Documentation


## 🔌 API Endpoints

### 1. Execute Data Science Workflow

Executes end-to-end data analysis, trains ML models inside the sandbox, and generates diagnostic plots.

* **POST** `/api/v1/execute`
* **Content-Type**: `multipart/form-data`

**Parameters:**

* `file` *(file, required)*: Input CSV dataset (e.g., `employee_records.csv`).
* `query` *(string, required)*: Target instruction (e.g., *"Train a model to predict Country using Department, Position, and Salary"*).

**Response:**

```json
{
  "status": "success",
  "execution_time_seconds": 12.4,
  "metrics": {
    "accuracy": 0.0998,
    "confusion_matrix_shape": [10, 10]
  },
  "artifacts": {
    "python_script": "/static/exports/pipeline.py",
    "jupyter_notebook": "/static/exports/pipeline.ipynb",
    "executive_report": "/static/exports/report.html"
  }
}

```

---

## 📖 Quick Start & Local Setup

### 1. Prerequisites

* Python `3.12+`
* `uv` (Fast Python Package Installer) or standard `pip`
* Docker & Docker Compose (optional for containerized run)
* **E2B API Key** (for sandbox code execution)
* **Groq / OpenAI / AWS Bedrock Key** (for LLM inference)

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# E2B Sandbox Key
E2B_API_KEY=your_e2b_api_key_here

# Service Config
PORT=8000
HOST=0.0.0.0

```

### 3. Local Installation (`uv` / `pip`)

```bash
# Clone the repository
git clone https://github.com/shamorai/Autonomous-AI-Data-Scientist.git
cd Autonomous-AI-Data-Scientist

# Create virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

```

### 4. Running the Application

```bash
# Start FastAPI Server
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

```

* **Interactive API Docs (Swagger)**: `http://localhost:8000/docs`
* **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## 🐳 Docker Deployment

To run the platform inside a production-ready container:

```bash
# Build Docker Image
docker build -t autonomous-ai-data-scientist .

# Run Docker Container
docker run -d -p 8000:8000 --env-file .env autonomous-ai-data-scientist

```

Or using **Docker Compose**:

```bash
docker-compose up --build -d

```

---

## 📚 Technology Stack

| Component | Technology | Description |
| --- | --- | --- |
| **Language** | Python 3.12 | Primary execution runtime |
| **Backend API** | FastAPI / Uvicorn | Async microservice framework |
| **Agent Framework** | LangChain / LangGraph | Workflow state machine & LLM orchestration |
| **Code Execution** | E2B MicroVM Sandbox | Isolated cloud sandbox for dynamic code execution |
| **ML & Analytics** | Scikit-Learn, SHAP, Pandas | Feature engineering, AutoML, and model explainability |
| **Containerization** | Docker / Render | Cloud deployment & container management |

---

## 🤝 Contributing

1. Fork the Project Repository
2. Create your Feature Branch (`git checkout -b feature/AwesomeFeature`)
3. Commit your Changes (`git commit -m 'feat: Add AwesomeFeature'`)
4. Push to the Branch (`git push origin feature/AwesomeFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 💬 Author & Contact

**Dev Upadhyay**

*AI Engineer & Data Scientist*

* 🐙 **GitHub**: [@shamorai](https://www.google.com/search?q=https://github.com/shamorai)
* 💼 **LinkedIn**: [Dev Upadhyay](https://www.google.com/search?q=https://www.linkedin.com/in/dev-upadhyay-9966002a1/)
