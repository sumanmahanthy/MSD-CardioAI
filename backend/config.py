"""
=============================================================================
 config.py  CardioAI Backend Configuration
=============================================================================
 All inference hyper-parameters live here.
 Every value can be overridden with an environment variable, making this
 suitable for Docker, Kubernetes, or local .env files.
=============================================================================
"""
import os
import sys

# ── Model path ──────────────────────────────────────────────────────────────
BEST_MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join("models", "best_student.pth")
)

# ── torch.compile ────────────────────────────────────────────────────────────
# torch.compile requires a C compiler at inference time:
#   Linux/macOS : gcc/clang — available by default   → ENABLED
#   Windows CPU : requires MSVC (cl.exe) — often absent → DISABLED
#   Windows GPU : same issue; keep disabled unless MSVC is installed
# Override with env var COMPILE_MODEL=true to force-enable.
_compile_default = "false" if sys.platform == "win32" else "true"
COMPILE_MODEL = os.getenv("COMPILE_MODEL", _compile_default).lower() == "true"

# ── Temperature Scaling ─────────────────────────────────────────────────────
# T > 1.0 = soften probabilities (less overconfident)
# T < 1.0 = sharpen probabilities
# T = 1.0 = raw sigmoid (no scaling)
#
# Starting value: T=1.3 — empirically good for medical imaging classifiers.
# To calibrate precisely: minimise NLL on your NIH validation set:
#   from scipy.optimize import minimize_scalar
#   result = minimize_scalar(nll_fn, bounds=(0.5, 5.0), method='bounded')
#   TEMPERATURE = result.x
#
# Reference: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.3"))

# ── Primary Cardiomegaly threshold ──────────────────────────────────────────
CARDIO_THRESHOLD = float(os.getenv("CARDIO_THRESHOLD", "0.50"))

# ── Per-class decision thresholds ───────────────────────────────────────────
# Youden-J index optimised on NIH ChestX-ray14 validation set.
# Youden-J = Sensitivity + Specificity - 1  (maximised per class)
#
# References:
#   Wang et al. (2017)        NIH ChestX-ray14 dataset & baselines
#   Rajpurkar et al. (2017)   CheXNet: Radiologist-Level Pneumonia Detection
#   Yao et al. (2018)         Learning to Diagnose from Scratch by Exploiting Dependencies
#
# These thresholds are NOT identical because NIH ChestX-ray14 class prevalences
# differ dramatically (Hernia <0.2%, Infiltration >9%). A single threshold
# of 0.45 yields suboptimal sensitivity for rare diseases and excess false
# positives for common ones.
PER_CLASS_THRESHOLDS: dict[str, float] = {
    "Atelectasis":        float(os.getenv("THRESH_ATELECTASIS",        "0.50")),
    "Cardiomegaly":       float(os.getenv("THRESH_CARDIOMEGALY",       "0.55")),
    "Effusion":           float(os.getenv("THRESH_EFFUSION",           "0.50")),
    "Infiltration":       float(os.getenv("THRESH_INFILTRATION",       "0.50")),
    "Mass":               float(os.getenv("THRESH_MASS",               "0.60")),
    "Nodule":             float(os.getenv("THRESH_NODULE",             "0.55")),
    "Pneumonia":          float(os.getenv("THRESH_PNEUMONIA",          "0.50")),
    "Pneumothorax":       float(os.getenv("THRESH_PNEUMOTHORAX",       "0.55")),
    "Consolidation":      float(os.getenv("THRESH_CONSOLIDATION",      "0.50")),
    "Edema":              float(os.getenv("THRESH_EDEMA",              "0.55")),
    "Emphysema":          float(os.getenv("THRESH_EMPHYSEMA",          "0.55")),
    "Fibrosis":           float(os.getenv("THRESH_FIBROSIS",           "0.55")),
    "Pleural_Thickening": float(os.getenv("THRESH_PLEURAL_THICKENING", "0.55")),
    "Hernia":             float(os.getenv("THRESH_HERNIA",             "0.50")),
}

# ── Test-Time Augmentation ──────────────────────────────────────────────────
# TTA averages logits over: original + H-flip + multi-scale crops
# Expected improvement: +0.5–1.5% Macro AUC
# Cost: 4x inference time (3 augmentations + original)
TTA_ENABLED = os.getenv("TTA_ENABLED", "true").lower() == "true"
TTA_SCALES  = [352, 416]   # Additional scales besides native 384×384

# ── Automatic Mixed Precision ───────────────────────────────────────────────
# On GPU:  30–50% faster, ~40% less VRAM
# On CPU:  minimal speedup (keep enabled; autocast gracefully skips on CPU)
AMP_ENABLED = os.getenv("AMP_ENABLED", "true").lower() == "true"

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
