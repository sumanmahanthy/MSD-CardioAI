"""
=============================================================================
 app.py  CardioAI HuggingFace Spaces Entry Point
=============================================================================
 This file bridges FastAPI (backend) with Gradio (HuggingFace Spaces SDK).
 Gradio acts as the wrapper; all FastAPI routes remain fully accessible.

 Public API endpoints (called by Vercel React frontend):
   POST /predict              - Main X-ray inference
   GET  /dashboard-stats      - Model performance stats
   GET  /analytics            - Disease distribution analytics
   GET  /roc-curve            - ROC curve data
   GET  /confusion-matrix     - Confusion matrix data
   GET  /prediction-history   - Past predictions
   GET  /prediction-distribution - 3-way distribution

 HuggingFace Spaces serves this on port 7860 by default.
=============================================================================
"""

import sys
import os

# ── HuggingFace Spaces GPU decorator (required even on CPU tier) ──────────────
# The `spaces` package auto-installed by HF raises a runtime error if no
# @spaces.GPU decorated function exists. This dummy function satisfies it.
try:
    import spaces

    @spaces.GPU(duration=0)
    def _gpu_placeholder():
        """Dummy — satisfies spaces GPU detection. Inference runs on CPU."""
        pass

except Exception:
    # Not running on HuggingFace Spaces (e.g. local dev) — skip silently
    pass

# ── Add backend directory to Python path ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# ── Change working directory to backend (so relative paths work) ─────────────
os.chdir(os.path.join(os.path.dirname(__file__), "backend"))

# ── Import existing FastAPI app ───────────────────────────────────────────────
from main import app as fastapi_app  # noqa: E402

# ── Minimal Gradio UI (status page at root "/") ───────────────────────────────
import gradio as gr  # noqa: E402

with gr.Blocks(
    title="CardioAI Backend API",
) as demo:
    gr.Markdown("""
    # 🫀 CardioAI — Backend API
    **Triple-Teacher Knowledge Distillation | ConvNeXt-V2-Tiny Student**

    This HuggingFace Space hosts the FastAPI inference backend for CardioAI.
    The React frontend (on Vercel) communicates with this Space via REST API.

    ---
    ## Available Endpoints
    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | `POST` | `/predict` | Upload chest X-ray → get AI diagnosis |
    | `GET`  | `/dashboard-stats` | Model performance statistics |
    | `GET`  | `/analytics` | Disease distribution data |
    | `GET`  | `/roc-curve` | ROC curve (AUC = 0.9379) |
    | `GET`  | `/confusion-matrix` | Cardiomegaly confusion matrix |
    | `GET`  | `/prediction-history` | Past prediction history |
    | `GET`  | `/prediction-distribution` | 3-way prediction distribution |

    ---
    > **Model**: ConvNeXt-V2-Tiny (27.9M params) — Knowledge Distilled from
    > ConvNeXt-V2-Base + Swin-Base + BioMedCLIP teachers
    """)

# ── Mount Gradio UI on FastAPI app ────────────────────────────────────────────
# All FastAPI routes remain at their original paths.
# Gradio UI is available at "/gradio" path.
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")
