# ================================================================
# CARDIOAI V7 — FINAL ENSEMBLE: Per-Class Learned Weight Search
# Input:  preds_A_val.npy, preds_B_val.npy, preds_C_val.npy
#         preds_A_test.npy, preds_B_test.npy, preds_C_test.npy
#         Y_val.npy, Y_test.npy
# Output: Final macro AUC, per-class AUC, F1, booster_metrics.json
#
# Strategy:
#   Step 1 — Load all .npy files (add to Kaggle dataset after student runs)
#   Step 2 — Per-class weight grid search on val set (wA, wB, wC)
#   Step 3 — Per-class threshold optimization on val set
#   Step 4 — Apply to blind test set → final metrics
#   Step 5 — Export report JSON
# ================================================================

import os, json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from itertools import product

# ────────────────────────────────────────────────────────────────
# 1. PATHS — adjust if .npy files live in a Kaggle input dataset
# ────────────────────────────────────────────────────────────────
NPY_DIR  = "/kaggle/input/mymodels/"   # where you uploaded the .npy files
OUT_DIR  = "/kaggle/working/"

DISEASES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural_Thickening","Hernia"
]
NUM_CLASSES = len(DISEASES)

# ────────────────────────────────────────────────────────────────
# 2. LOAD PREDICTIONS
# ────────────────────────────────────────────────────────────────
print("Loading predictions...")

def load(tag, split):
    path = os.path.join(NPY_DIR, f"preds_{tag}_{split}.npy")
    arr  = np.load(path).astype(np.float32)
    print(f"  preds_{tag}_{split}: {arr.shape}")
    return arr

preds_A_val  = load("A", "val")
preds_B_val  = load("B", "val")
preds_C_val  = load("C", "val")
Y_val        = np.load(os.path.join(NPY_DIR, "Y_val.npy")).astype(np.float32)

preds_A_test = load("A", "test")
preds_B_test = load("B", "test")
preds_C_test = load("C", "test")
Y_test       = np.load(os.path.join(NPY_DIR, "Y_test.npy")).astype(np.float32)

print(f"\nVal  set: {Y_val.shape[0]:,} samples")
print(f"Test set: {Y_test.shape[0]:,} samples")

# Quick sanity: each model's solo val AUC
for tag, p in [("A", preds_A_val), ("B", preds_B_val), ("C", preds_C_val)]:
    print(f"  Student-{tag} solo Val AUC: {roc_auc_score(Y_val, p, average='macro'):.4f}")

# ────────────────────────────────────────────────────────────────
# 3. SIMPLE EQUAL-WEIGHT BASELINE
# ────────────────────────────────────────────────────────────────
equal_blend_val  = (preds_A_val  + preds_B_val  + preds_C_val)  / 3.0
equal_blend_test = (preds_A_test + preds_B_test + preds_C_test) / 3.0
equal_val_auc  = roc_auc_score(Y_val,  equal_blend_val,  average="macro")
equal_test_auc = roc_auc_score(Y_test, equal_blend_test, average="macro")
print(f"\nBaseline equal-weight Val  AUC: {equal_val_auc:.4f}")
print(f"Baseline equal-weight Test AUC: {equal_test_auc:.4f}")

# ────────────────────────────────────────────────────────────────
# 4. PER-CLASS WEIGHT GRID SEARCH ON VALIDATION SET
# Resolution: 0.1 step (11^2 combinations per class = fast)
# For finer resolution, change STEP to 0.05 (takes longer)
# ────────────────────────────────────────────────────────────────
STEP = 0.1
weights_grid = [round(x, 2) for x in np.arange(0.0, 1.0 + STEP, STEP)]

print(f"\nSearching per-class weights (step={STEP})...")
best_weights = np.zeros((NUM_CLASSES, 3))   # [class, (wA, wB, wC)]
best_class_aucs_val = np.zeros(NUM_CLASSES)

for c in range(NUM_CLASSES):
    best_auc, best_w = 0.0, (1/3, 1/3, 1/3)
    y_c = Y_val[:, c]

    for wA in weights_grid:
        for wB in weights_grid:
            wC = round(1.0 - wA - wB, 4)
            if wC < 0 or wC > 1.0: continue

            blend = wA * preds_A_val[:, c] + wB * preds_B_val[:, c] + wC * preds_C_val[:, c]
            try:
                auc = roc_auc_score(y_c, blend)
            except ValueError:
                continue
            if auc > best_auc:
                best_auc, best_w = auc, (wA, wB, wC)

    best_weights[c]        = best_w
    best_class_aucs_val[c] = best_auc
    wA, wB, wC = best_w
    print(f"  {DISEASES[c]:<22} Val AUC: {best_auc:.4f}  w=({wA:.1f},{wB:.1f},{wC:.1f})")

# Build per-class weighted val blend
blend_val = np.zeros_like(preds_A_val)
for c in range(NUM_CLASSES):
    wA, wB, wC = best_weights[c]
    blend_val[:, c] = wA*preds_A_val[:,c] + wB*preds_B_val[:,c] + wC*preds_C_val[:,c]

learned_val_auc = float(roc_auc_score(Y_val, blend_val, average="macro"))
print(f"\nLearned per-class Val AUC:      {learned_val_auc:.4f}  (vs equal-weight {equal_val_auc:.4f})")

# ────────────────────────────────────────────────────────────────
# 5. PER-CLASS THRESHOLD OPTIMIZATION ON VALIDATION SET
# ────────────────────────────────────────────────────────────────
print("\nOptimizing per-class decision thresholds on validation set...")
best_thresholds = np.zeros(NUM_CLASSES)
best_val_f1s    = np.zeros(NUM_CLASSES)

for c in range(NUM_CLASSES):
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.05, 0.95, 0.05):
        preds_binary = (blend_val[:, c] >= th).astype(int)
        f1 = f1_score(Y_val[:, c].astype(int), preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    best_thresholds[c] = best_th
    best_val_f1s[c]    = best_f1

print(f"  Threshold range: [{best_thresholds.min():.2f}, {best_thresholds.max():.2f}]")
print(f"  Mean val F1 after threshold opt: {best_val_f1s.mean():.4f}")

# ────────────────────────────────────────────────────────────────
# 6. APPLY TO BLIND TEST SET
# ────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL BLIND TEST SET EVALUATION")
print("="*55)

blend_test = np.zeros_like(preds_A_test)
for c in range(NUM_CLASSES):
    wA, wB, wC = best_weights[c]
    blend_test[:, c] = wA*preds_A_test[:,c] + wB*preds_B_test[:,c] + wC*preds_C_test[:,c]

final_macro_auc = float(roc_auc_score(Y_test, blend_test, average="macro"))

per_class_auc = {}
per_class_f1  = {}
for c in range(NUM_CLASSES):
    try:
        auc_c = float(roc_auc_score(Y_test[:, c], blend_test[:, c]))
    except ValueError:
        auc_c = float("nan")
    f1_c = float(f1_score(Y_test[:, c].astype(int),
                           (blend_test[:, c] >= best_thresholds[c]).astype(int),
                           zero_division=0))
    per_class_auc[DISEASES[c]] = round(auc_c, 4)
    per_class_f1[DISEASES[c]]  = round(f1_c,  4)

macro_f1 = float(np.nanmean(list(per_class_f1.values())))

print(f"\n{'Disease':<24} {'AUC':>7}  {'F1':>6}  {'Thresh':>6}")
print("-" * 50)
for c, d in enumerate(DISEASES):
    print(f"  {d:<22} {per_class_auc[d]:>7.4f}  {per_class_f1[d]:>6.4f}  {best_thresholds[c]:>6.2f}")
print("-" * 50)
print(f"  {'MACRO AVERAGE':<22} {final_macro_auc:>7.4f}  {macro_f1:>6.4f}")
print(f"\n{'='*55}")
print(f"  FINAL ENSEMBLE MACRO AUC : {final_macro_auc:.4f}")
print(f"  FINAL ENSEMBLE MACRO F1  : {macro_f1:.4f}")
print(f"{'='*55}")

if final_macro_auc >= 0.93:
    print("\n  TARGET 0.93+ ACHIEVED!")
elif final_macro_auc >= 0.90:
    print(f"\n  Strong result. Gap to 0.93: {0.93 - final_macro_auc:.4f}")
else:
    print(f"\n  Gap to 0.93: {0.93 - final_macro_auc:.4f}")

# ────────────────────────────────────────────────────────────────
# 7. EXPORT REPORT JSON
# ────────────────────────────────────────────────────────────────
report = {
    "Final_Macro_AUC":         final_macro_auc,
    "Final_Macro_F1":          macro_f1,
    "Baseline_EqualWeight_AUC": equal_test_auc,
    "Learned_Val_AUC":         learned_val_auc,
    "Per_Class_AUC":           per_class_auc,
    "Per_Class_F1":            per_class_f1,
    "Optimal_Weights": {
        DISEASES[c]: {
            "wA": float(round(best_weights[c][0], 2)),
            "wB": float(round(best_weights[c][1], 2)),
            "wC": float(round(best_weights[c][2], 2)),
        } for c in range(NUM_CLASSES)
    },
    "Optimal_Thresholds": {DISEASES[c]: float(best_thresholds[c]) for c in range(NUM_CLASSES)},
}

out_path = os.path.join(OUT_DIR, "v7_ensemble_report.json")
with open(out_path, "w") as f:
    json.dump(report, f, indent=4)
print(f"\nFull report saved → v7_ensemble_report.json")
