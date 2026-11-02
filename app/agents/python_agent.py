import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.schemas.profile import DatasetProfile
from app.schemas.agent import PythonAgentOutput


class PythonCodeAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(
            temperature=0.1,
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY")
        )



        system_instruction = (
    "You are an expert Data Scientist. Write executable Python code for DataFrame `df`.\n\n"
    "MANDATORY STRICT RULES:\n"
    "1. DataFrame `df` is PRE-LOADED in memory as `df`. NEVER re-read CSV files.\n"
    "2. DO NOT run `pip install` inside the python script. All packages are pre-installed.\n"
    "3. DROP HIGH-CARDINALITY TEXT/ID COLUMNS before training:\n"
    "   `drop_cols = [c for c in ['Employee_Name', 'Employee_ID', 'ID', 'Name'] if c in df.columns]; X = df.drop(columns=drop_cols + ['Department'])`\n"
    "4. MANDATORY METRICS PRINTING TO STDOUT:\n"
    "   - ALWAYS print model metrics directly using print() so E2B Sandbox captures stdout:\n"
    "     print(f'MODEL_ACCURACY:{{acc:.4f}}')\n"
    "     print('CONFUSION_MATRIX:')\n"
    "     print(cm)\n"
    "     print('CLASSIFICATION_REPORT:')\n"
    "     print(classification_report(y_test, y_pred))\n"
    "5. MANDATORY BASE64 IMAGE STDOUT (APPLIES TO ALL PLOTS - CONFUSION MATRIX & SHAP):\n"
    "   - Set non-interactive backend at top: `import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt`.\n"
    "   - EVERY plot figure MUST be saved to a BytesIO buffer, encoded to Base64, and printed to stdout using exact 'IMAGE_BASE64:' prefix:\n"
    "     import io, base64\n"
    "     buf = io.BytesIO()\n"
    "     plt.savefig(buf, format='png', bbox_inches='tight', dpi=70)\n"
    "     buf.seek(0)\n"
    "     img_b64 = base64.b64encode(buf.read()).decode('utf-8').replace('\\n', '').strip()\n"
    "     print('IMAGE_BASE64:' + img_b64)\n"
    "     plt.clf()\n"
    "     plt.close('all')\n"
    "6. SAFE & GUARANTEED SHAP SUMMARY PLOT:\n"
    "   - Sample max 15 rows: `X_sample = X_test.sample(n=min(15, len(X_test)), random_state=42)`\n"
    "   - Preprocess sample: `X_sample_prep = model.named_steps['preprocess'].transform(X_sample)`\n"
    "   - Convert to dense numpy array: `if hasattr(X_sample_prep, 'toarray'): X_sample_prep = X_sample_prep.toarray()`\n"
    "   - Safely extract feature names:\n"
    "     try:\n"
    "         feat_names = list(model.named_steps['preprocess'].get_feature_names_out())\n"
    "     except Exception:\n"
    "         feat_names = [f'feature_{i}' for i in range(X_sample_prep.shape[1])]\n"
    "   - Compute SHAP & slice for Multiclass safely:\n"
    "     explainer = shap.TreeExplainer(model.named_steps['clf'])\n"
    "     shap_vals = explainer.shap_values(X_sample_prep)\n"
    "     if isinstance(shap_vals, list):\n"
    "         shap_vals = shap_vals[0]\n"
    "     elif len(np.array(shap_vals).shape) == 3:\n"
    "         shap_vals = shap_vals[:, :, 0]\n"
    "   - `plt.figure(figsize=(8, 4))`\n"
    "   - `shap.summary_plot(shap_vals, X_sample_prep, feature_names=feat_names, plot_type='bar', show=False)`\n"
    "   - Save SHAP plot to buffer and print with `IMAGE_BASE64:` prefix as specified in Rule 5.\n"
    "7. MODEL SERIALIZATION:\n"
    "   - ALWAYS save fitted model using `import joblib; joblib.dump(model, 'model.joblib')`.\n"
    "8. Return raw Python script text only. Do NOT wrap inside markdown."
) 

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", 
             "Question: {question}\n"
             "Dataset Summary:\n{profile}\n"
             "Error Context: {error_context}")
        ])

        self.chain = self.prompt | self.llm

    def generate_code(self, question: str, profile: str, error_context: str = "None") -> str:
        res = self.chain.invoke({
            "question": str(question), 
            "profile": str(profile), 
            "error_context": str(error_context),
            "i": 0
        })
        clean_code = res.content.replace("```python", "").replace("```", "").strip()
        return clean_code


PythonAnalyticalAgent = PythonCodeAgent