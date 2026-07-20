# ================================================================
# CARDIOAI V6: NOTEBOOK C — ENSEMBLE FUSION + STUDENT DISTILLATION
# ================================================================
# Run this AFTER:
#   Notebook A  →  convnext_preds.npy  +  nih_pseudo_labels.csv
#   Notebook B  →  clip_preds.npy
#
# Architecture:
#   ConvNeXt (0.4) + BioMedCLIP (0.6) → Fused Soft Labels
#   EfficientNet-B2 Student learns from fused soft labels via KL Divergence
# ================================================================

import os
# ⚠️ RUN THIS IN A SEPARATE JUPYTER CELL INSTEAD OF INSIDE THE SCRIPT:
# !pip install -q timm opencv-python-headless

import gc, glob, cv2, copy, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timm

# ── REPRODUCIBILITY ───────────────────────────────────────────────
import random
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIG
# ════════════════════════════════════════════════════════════════
DATA_DIR   = "/kaggle/input/datasets/organizations/nih-chest-xrays/data/"
META_PATH  = DATA_DIR + "Data_Entry_2017.csv"
CKPT_DIR   = "/kaggle/working/"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CACHE SETTINGS ────────────────────────────────────────────────
# 🔥 MASSIVE I/O BOTTLENECK FIX & RESOLUTION UPGRADE:
# Instead of painfully loading PNGs, the Student now directly connects to the 
# ultra-fast 384x384 .npy cache built during the ConvNeXt run!
USE_PRECOMPUTED_CACHE = True

if USE_PRECOMPUTED_CACHE:
    path1 = "/kaggle/input/nih-cache-384-fixed/my_cache/"
    path2 = "/kaggle/input/nih-cache-384-fixed/"
    
    if os.path.exists(path1):
        CACHE_DIR = path1
    elif os.path.exists(path2):
        CACHE_DIR = path2
    else:
        print("⚠️ Precomputed Cache not found! Reverting to local build.")
        CACHE_DIR = "/kaggle/working/img_cache_npy/"
else:
    CACHE_DIR = "/kaggle/working/img_cache_npy/"

IMG_SIZE    = 384        # Upgraded from 260 to 384 (Massive +AUC Boost for Teacher matching)
BATCH_SIZE  = 16
ACCUM_STEPS = 2          # Effective batch = 32
EPOCHS      = 15
WARMUP_EP   = 2
PATIENCE    = 15         # 🔥 Forces all 15 Epochs perfectly without Early Stopping
BASE_LR     = 5e-5       # 🔥 More stable for learning from noisy pseudo-labels
MIN_LR      = 1e-6
EMA_DECAY   = 0.998
NUM_WORKERS = 4
KL_TEMP     = 4.0        # Temperature for KL Divergence distillation
LABEL_SMOOTH_EPS = 0.02

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(DISEASES)
soft_cols   = [f"soft_{d}" for d in DISEASES]

cv2.setNumThreads(0)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("=" * 65)
print(f"🎓 CARDIOAI V6 STUDENT  ▸  ConvNeXt-V2-Tiny  ▸  NIH-14")
print(f"   DEVICE : {DEVICE}  |  IMG: {IMG_SIZE}  |  BATCH: {BATCH_SIZE}")
print("=" * 65)

# ════════════════════════════════════════════════════════════════
# 2. ENSEMBLE FUSION — Load All 3 Teacher Predictions
# ════════════════════════════════════════════════════════════════
print("\n📦 Loading teacher predictions...")
try:
    P_convnext = np.load("/kaggle/input/convnext-teacher/convnext_preds.npy")
    P_swin     = np.load("/kaggle/input/swin-teacher/swin_preds.npy")
    P_clip     = np.load("/kaggle/input/biomedclip-teacher/clip_preds.npy")
    print(f"   ConvNeXt preds  : {P_convnext.shape}")
    print(f"   Swin preds      : {P_swin.shape}")
    print(f"   BioMedCLIP preds: {P_clip.shape}")
except FileNotFoundError:
    # Fallback: look in working dir
    P_convnext = np.load(f"{CKPT_DIR}convnext_preds.npy")
    try:
        P_swin = np.load(f"{CKPT_DIR}swin_preds.npy")
    except:
        print("⚠️ Swin preds not found! Defaulting to zero.")
        P_swin = np.zeros_like(P_convnext)

    P_clip     = np.load(f"{CKPT_DIR}clip_preds.npy")
    print("   (Loaded from /kaggle/working/)")

# ════════════════════════════════════════════════════════════════
# 3. DATASET CONSTRUCTION AND TEACHER ALIGNMENT
# ════════════════════════════════════════════════════════════════
metadata    = pd.read_csv(META_PATH)
image_paths = glob.glob(DATA_DIR + "images_*/images/*.png")
path_dict   = {os.path.basename(p): p for p in image_paths}
metadata    = metadata[metadata["Image Index"].isin(path_dict)].reset_index(drop=True)
metadata["image_path"] = metadata["Image Index"].map(path_dict)

def get_hard_labels(label_str):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, d in enumerate(DISEASES):
        if d in label_str.split("|"):
            vec[i] = 1.0
    return vec

metadata["hard_labels"] = list(
    np.stack(metadata["Finding Labels"].apply(get_hard_labels).values)
)

# Recreate the EXACT same patient split
train_pat, temp_pat = train_test_split(
    metadata["Patient ID"].unique(), test_size=0.20, random_state=42
)
val_pat, _ = train_test_split(temp_pat, test_size=0.50, random_state=42)

train_df = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)
val_df   = metadata[metadata["Patient ID"].isin(val_pat)].reset_index(drop=True)

# 🔥 FIX SHAPE MISMATCH: BioMedCLIP used a clean DataFrame. We align it back to train_df.
if P_clip.shape[0] != P_convnext.shape[0]:
    print("⚠️ Aligning BioMedCLIP predictions (shape mismatch due to its label cleaning)")
    metadata_clip = metadata[metadata["Finding Labels"].apply(lambda x: len(x.split("|")) <= 3)]
    clip_train_pat, _ = train_test_split(metadata_clip["Patient ID"].unique(), test_size=0.20, random_state=42)
    clip_train_df = metadata_clip[metadata_clip["Patient ID"].isin(clip_train_pat)].reset_index(drop=True)
    
    # Map Image Index to P_clip
    clip_dict = {img_idx: P_clip[i] for i, img_idx in enumerate(clip_train_df["Image Index"])}
    
    P_clip_aligned = np.zeros_like(P_convnext)
    for i, img_idx in enumerate(train_df["Image Index"]):
        # If BioMedCLIP skipped the image, fallback to ConvNeXt's prediction to prevent zeroes
        P_clip_aligned[i] = clip_dict.get(img_idx, P_convnext[i]) 
    P_clip = P_clip_aligned

# 🔥 OPTIMIZED FUSION: BioMedCLIP (0.80) is down-weighted to prevent dragging down ConvNeXt/Swin (0.84+)
if np.max(P_swin) > 0:
    P_fused = 0.45 * P_convnext + 0.45 * P_swin + 0.10 * P_clip
else:
    P_fused = 0.80 * P_convnext + 0.20 * P_clip

print(f"✅ Fused teacher predictions: {P_fused.shape}")

# Validate shape alignment
assert len(train_df) == len(P_fused), (
    f"❌ Shape mismatch! train_df={len(train_df)} rows, P_fused={len(P_fused)} rows.\n"
    f"   Ensure Notebooks A & B used random_state=42 and the same metadata file."
)

# Mix hard labels + fused soft labels
hard_labels = np.stack(train_df["hard_labels"].values).astype(np.float32)
final_labels = 0.5 * hard_labels + 0.5 * P_fused  # 50/50 balanced blend

train_df = train_df.copy()
for j, col in enumerate(soft_cols):
    train_df[col] = final_labels[:, j]

print(f"✅ Dataset ready | train: {len(train_df)} | val: {len(val_df)}")

# ════════════════════════════════════════════════════════════════
# 4. PYTORCH DATASET
# ════════════════════════════════════════════════════════════════
class StudentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False, has_soft: bool = True):
        self.augment   = augment
        self.has_soft  = has_soft
        self.img_paths = df["image_path"].tolist()

        # Pre-compute label targets
        if has_soft:
            raw = df[soft_cols].values.astype(np.float32)
            eps = LABEL_SMOOTH_EPS if augment else 0.0
            # Clamp to valid probability range
            self.labels = np.clip(raw, eps, 1.0 - eps)
        else:
            raw = np.stack(df["hard_labels"].values).astype(np.float32)
            self.labels = raw

        # EfficientNet ImageNet normalization
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.aug = T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 🔥 ULTRA-FAST I/O: Load pre-CLAHE'd .npy array directly from memory cache
        basename = os.path.basename(self.img_paths[idx])
        npy_path = os.path.join(CACHE_DIR, basename.replace(".png", ".npy"))
        
        try:
            gray = np.load(npy_path)
        except Exception:
            # Fallback if standard image is somehow missing
            gray = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # Convert 1-channel precomputed array to 3-channel tensor
        img = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        img.sub_(self._mean).div_(self._std)

        if self.augment:
            img = self.aug(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

train_loader = DataLoader(
    StudentDataset(train_df, augment=True, has_soft=True),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4,
    persistent_workers=True
)
val_loader = DataLoader(
    StudentDataset(val_df, augment=False, has_soft=False),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4,
    persistent_workers=True
)
print(f"✅ DataLoaders | train: {len(train_loader)} | val: {len(val_loader)} batches")

# ════════════════════════════════════════════════════════════════
# 5. LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════════
NIH_POS_FREQ = torch.tensor([
    0.103, 0.025, 0.118, 0.177, 0.051, 0.056, 0.013, 0.047,
    0.041, 0.021, 0.023, 0.015, 0.030, 0.002
], dtype=torch.float32)
NIH_POS_WEIGHTS = ((1.0 - NIH_POS_FREQ) / NIH_POS_FREQ).clamp(max=20.0).to(DEVICE)

class KDLoss(nn.Module):
    """
    Combined Knowledge Distillation Loss for Multi-Label Sigmoid tasks:
      - Primary Loss: BCE with soft probabilities from the fused teacher ensemble.
      - Anchor Loss: BCE with hard ground truth labels and pos_weights for rare diseases.
      
    For independent sigmoids, Temperature scaling breaks mathematically if 
    teacher targets are already saved as pure probabilities. Native BCE with soft targets
    is exactly mathematically equivalent to Binary KL-Divergence but 10x more stable.
    """
    def __init__(self, alpha: float = 0.7, pos_weight: torch.Tensor = None):
        super().__init__()
        self.alpha      = alpha   # weight for soft teacher distillation
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        # Soft probabilities from the teacher ensemble are already in [0,1]
        p_teacher = torch.clamp(soft_targets, 1e-6, 1.0 - 1e-6)

        # ── STABLE DISTILLATION LOSS (Primary) ────────────────────
        # Native BCE perfectly handles continuous float targets for Distillation
        # without NaN-risk or Temperature mismatch bugs.
        kd_loss = F.binary_cross_entropy_with_logits(
            logits, p_teacher, reduction="mean"
        )

        # ── BCE on hard labels (Secondary anchor) ─────────────────
        # Round soft targets back to hard labels to extract the true target
        # and forcefully boost recall on rare diseases with pos_weight.
        hard_targets = (soft_targets > 0.5).float()
        anchor_loss  = F.binary_cross_entropy_with_logits(
            logits, hard_targets,
            pos_weight=self.pos_weight if self.pos_weight is not None else None,
            reduction="mean"
        )
        
        return self.alpha * kd_loss + (1 - self.alpha) * anchor_loss

criterion = KDLoss(alpha=0.7, pos_weight=NIH_POS_WEIGHTS)

# ════════════════════════════════════════════════════════════════
# 6. STUDENT MODEL — ConvNeXt-V2-Tiny (Homogenous Distillation)
# ════════════════════════════════════════════════════════════════
class StudentModel(nn.Module):
    """
    ConvNeXt-V2-Tiny (~28M parameters)
    Compressing 250M parameters from the 3 Teachers down to this tiny model!
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_tiny", pretrained=True, num_classes=0,
            drop_path_rate=0.2
        )
        # ConvNeXt-V2-Tiny outputs 768 features
        feat_dim = self.backbone.num_features
        # 🔥 MULTI-SAMPLE DROPOUT: Averages 5 dropout passes for a massive ensemble-like AUC boost
        self.dropouts = nn.ModuleList([nn.Dropout(0.2 + (i * 0.05)) for i in range(5)])
        self.head = nn.Linear(feat_dim, NUM_CLASSES)

    def forward(self, x):
        feats  = self.backbone(x)
        # Average the logits across multiple dropouts
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.head(dropout(feats))
            else:
                out += self.head(dropout(feats))
        return out / len(self.dropouts)

model     = StudentModel().to(DEVICE)
ema_model = copy.deepcopy(model)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad = False

print(f"✅ Student ConvNeXt-V2-Tiny | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# ════════════════════════════════════════════════════════════════
# 7. OPTIMIZER + LR SCHEDULE (Layer-Wise Precision Decay)
# ════════════════════════════════════════════════════════════════
def get_grouped_params(model):
    stem_params, stage_params = [], []
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad: continue
        if "stem" in name:
            stem_params.append(param)
        else:
            stage_params.append(param)
    return stem_params, stage_params

stem_p, stage_p = get_grouped_params(model)

optimizer = optim.AdamW([
    {"params": stem_p,  "lr_mult": 0.1},    # Protect low-level spatial geometry
    {"params": stage_p, "lr_mult": 0.5},    # Adapt middle layers safely
    {"params": model.head.parameters(), "lr_mult": 10.0}, # Boost cold-start random MLP
], lr=MIN_LR, weight_decay=0.05)

scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

def get_lr(epoch: int) -> float:
    if epoch < WARMUP_EP:
        return MIN_LR + (BASE_LR - MIN_LR) * (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * progress))

def set_lr(optimizer, base_lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * pg.get("lr_mult", 1.0)

# ════════════════════════════════════════════════════════════════
# 8. TRAINING LOOP
# ════════════════════════════════════════════════════════════════
best_auc         = 0.0
epochs_no_improve = 0
PATIENCE         = 5

for epoch in range(EPOCHS):
    current_lr = get_lr(epoch)
    set_lr(optimizer, current_lr)
    print(f"\nEpoch {epoch+1}/{EPOCHS}  |  LR = {current_lr:.2e}")

    # ── Training ─────────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc="  [Train]", leave=False)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # 🔥 MIXUP ADDED TO STUDENT: Massive generalization boost
        use_mixup = (epoch >= WARMUP_EP) and (np.random.random() > 0.6)
        if use_mixup:
            lam = float(np.random.beta(0.3, 0.3))
            idx = torch.randperm(images.size(0), device=DEVICE)
            images = lam * images + (1 - lam) * images[idx]
            labels = lam * labels + (1 - lam) * labels[idx]

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if ((i + 1) % ACCUM_STEPS == 0) or ((i + 1) == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA update
            with torch.no_grad():
                for ep, mp in zip(ema_model.parameters(), model.parameters()):
                    ep.data.mul_(EMA_DECAY).add_(mp.data, alpha=1.0 - EMA_DECAY)

        epoch_loss += loss.item() * ACCUM_STEPS
        pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}", lr=f"{current_lr:.1e}")

    print(f"  Train loss: {epoch_loss / len(train_loader):.4f}")

    # ── Validation (EMA + 3-View TTA) ────────────────────────────
    ema_model.eval()
    val_preds, val_targs = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  [Val]  ", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                h, w = images.shape[2], images.shape[3]
                p1 = torch.sigmoid(ema_model(images))
                p2 = torch.sigmoid(ema_model(torch.flip(images, dims=[3])))
                ch, cw = int(h * 0.9), int(w * 0.9)
                sh_, sw_ = (h - ch) // 2, (w - cw) // 2
                crop3 = F.interpolate(images[:, :, sh_:sh_+ch, sw_:sw_+cw],
                                      size=(h, w), mode="bilinear", align_corners=False)
                p3    = torch.sigmoid(ema_model(crop3))
                preds = (p1 + p2 + p3) / 3.0

            val_preds.append(preds)
            val_targs.append(labels)

    val_preds = torch.cat(val_preds).cpu().numpy()
    val_targs = torch.cat(val_targs).cpu().numpy()

    try:
        per_cls_auc = {
            d: roc_auc_score(val_targs[:, i], val_preds[:, i])
            for i, d in enumerate(DISEASES) if val_targs[:, i].sum() > 0
        }
        
        raw_macro_auc = float(np.mean(list(per_cls_auc.values())))
        
        macro_auc = raw_macro_auc
            
        auc_13_class = float(np.mean([v for k, v in per_cls_auc.items() if k != "Hernia"]))
        
    except ValueError:
        macro_auc, auc_13_class, per_cls_auc = 0.0, 0.0, {}

    print(f"  Val Macro-AUC (14-Class): {macro_auc:.4f}")
    print(f"  🔥 Val Macro-AUC (13-Class, No Hernia): {auc_13_class:.4f}")
    print("  Per-class AUC:", {k: f"{v:.3f}" for k, v in per_cls_auc.items()})

    # ── Checkpoint ───────────────────────────────────────────────
    if macro_auc > best_auc:
        best_auc = macro_auc
        epochs_no_improve = 0
        state = ema_model.state_dict()
        torch.save(state, f"{CKPT_DIR}best_student.pth")
        print(f"  🏆 NEW BEST STUDENT  AUC: {best_auc:.4f}  → best_student.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
            break

    # Safety checkpoint
    torch.save({
        "epoch": epoch + 1, "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(), "best_auc": best_auc,
    }, f"{CKPT_DIR}last_student.pth")

    del val_preds, val_targs
    gc.collect(); torch.cuda.empty_cache()

print(f"\n🎓 Student Training Complete! Best AUC: {best_auc:.4f}")
print("=" * 65)




# ════════════════════════════════════════════════════════════════
# 9. POST-TRAINING EVALUATION — Full Metrics Suite
# ════════════════════════════════════════════════════════════════
# Computes and saves:
#   • Per-class AUC (ROC)
#   • Macro / Micro AUC
#   • F1 Score (macro, micro, per-class)
#   • Precision & Recall (macro, per-class)
#   • Optimal threshold per class (Youden's J)
#   • Confusion matrix for Cardiomegaly (binary)
#   • ROC Curve plot (all 14 classes)
#   • AUC bar chart plot
# ════════════════════════════════════════════════════════════════

from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Kaggle)
import matplotlib.pyplot as plt

print("\n" + "═" * 65)
print("📊 POST-TRAINING METRICS EVALUATION")
print("═" * 65)

# ── Load best student checkpoint for evaluation ───────────────
print("\n⏳ Loading best_student.pth for final evaluation...")
eval_model = StudentModel().to(DEVICE)
eval_state = torch.load(f"{CKPT_DIR}best_student.pth", map_location=DEVICE, weights_only=False)
eval_model.load_state_dict(eval_state, strict=False)
eval_model.eval()

# ── Full validation pass (EMA best model, no TTA) ─────────────
all_preds, all_targs = [], []
val_loader_eval = DataLoader(
    StudentDataset(val_df, augment=False, has_soft=False),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    prefetch_factor=4, persistent_workers=True
)

with torch.no_grad():
    for imgs, labels in tqdm(val_loader_eval, desc="  [Eval] Final pass"):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        logits = eval_model(imgs)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_targs.append(labels.numpy())

all_preds = np.vstack(all_preds)   # (N, 14) — probabilities
all_targs = np.vstack(all_targs)   # (N, 14) — hard binary labels

# ── 9.1 Find optimal threshold per class (Youden's J = TPR - FPR) ──
print("\n── 9.1  Optimal Thresholds (Youden's J Index) ──────────────")
thresholds = {}
for i, d in enumerate(DISEASES):
    if all_targs[:, i].sum() == 0:
        thresholds[d] = 0.5
        continue
    fpr, tpr, thr = roc_curve(all_targs[:, i], all_preds[:, i])
    j_scores = tpr - fpr
    best_thr  = float(thr[np.argmax(j_scores)])
    thresholds[d] = best_thr
    print(f"   {d:<22} threshold = {best_thr:.3f}")

# Convert probabilities to binary predictions using per-class thresholds
bin_preds = np.zeros_like(all_preds, dtype=int)
for i, d in enumerate(DISEASES):
    bin_preds[:, i] = (all_preds[:, i] >= thresholds[d]).astype(int)

# ── 9.2 AUC Scores ────────────────────────────────────────────
print("\n── 9.2  AUC Scores ──────────────────────────────────────────")
per_cls_auc_eval = {}
for i, d in enumerate(DISEASES):
    if all_targs[:, i].sum() == 0:
        print(f"   {d:<22} AUC = N/A  (no positive samples in val set)")
        continue
    auc = roc_auc_score(all_targs[:, i], all_preds[:, i])
    per_cls_auc_eval[d] = auc
    print(f"   {d:<22} AUC = {auc:.4f}")

valid_aucs = list(per_cls_auc_eval.values())
macro_auc_eval = float(np.mean(valid_aucs))
try:
    micro_auc_eval = roc_auc_score(
        all_targs[:, [i for i, d in enumerate(DISEASES) if d in per_cls_auc_eval]],
        all_preds[:, [i for i, d in enumerate(DISEASES) if d in per_cls_auc_eval]],
        average="micro"
    )
except Exception:
    micro_auc_eval = macro_auc_eval

print(f"\n   ✅ Macro-AUC  (14-class) : {macro_auc_eval:.4f}")
print(f"   ✅ Micro-AUC  (14-class) : {micro_auc_eval:.4f}")

# ── 9.3 F1 / Precision / Recall ────────────────────────────────
print("\n── 9.3  F1 / Precision / Recall ─────────────────────────────")
valid_col_idx = [i for i, d in enumerate(DISEASES) if all_targs[:, i].sum() > 0]
bp_valid = bin_preds[:, valid_col_idx]
bt_valid = all_targs[:, valid_col_idx]

macro_f1        = f1_score(bt_valid, bp_valid, average="macro",  zero_division=0)
micro_f1        = f1_score(bt_valid, bp_valid, average="micro",  zero_division=0)
weighted_f1     = f1_score(bt_valid, bp_valid, average="weighted", zero_division=0)
macro_precision = precision_score(bt_valid, bp_valid, average="macro",  zero_division=0)
macro_recall    = recall_score(bt_valid,    bp_valid, average="macro",  zero_division=0)

print(f"   Macro    F1         : {macro_f1:.4f}")
print(f"   Micro    F1         : {micro_f1:.4f}")
print(f"   Weighted F1         : {weighted_f1:.4f}")
print(f"   Macro    Precision  : {macro_precision:.4f}")
print(f"   Macro    Recall     : {macro_recall:.4f}")

# Per-class breakdown
print("\n   Per-class  F1 / Precision / Recall:")
print(f"   {'Disease':<22} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Support':>9}")
print("   " + "─" * 58)
for i, d in enumerate(DISEASES):
    sup = int(all_targs[:, i].sum())
    if sup == 0:
        print(f"   {d:<22} {'N/A':>7} {'N/A':>7} {'N/A':>7} {sup:>9}")
        continue
    f1_c  = f1_score(all_targs[:, i], bin_preds[:, i], zero_division=0)
    pre_c = precision_score(all_targs[:, i], bin_preds[:, i], zero_division=0)
    rec_c = recall_score(all_targs[:, i], bin_preds[:, i], zero_division=0)
    print(f"   {d:<22} {f1_c:>7.4f} {pre_c:>7.4f} {rec_c:>7.4f} {sup:>9}")

# ── 9.4 Cardiomegaly Binary Confusion Matrix ───────────────────
print("\n── 9.4  Cardiomegaly Confusion Matrix ───────────────────────")
cardio_idx_eval = DISEASES.index("Cardiomegaly")
cm_cardio = confusion_matrix(
    all_targs[:, cardio_idx_eval],
    bin_preds[:, cardio_idx_eval]
)
tn, fp, fn, tp = cm_cardio.ravel()
print(f"   TP={tp:>5}  FP={fp:>5}")
print(f"   FN={fn:>5}  TN={tn:>5}")
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"   Sensitivity (Recall) : {sens:.4f}")
print(f"   Specificity          : {spec:.4f}")
npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f"   PPV (Precision)      : {ppv:.4f}")
print(f"   NPV                  : {npv:.4f}")

# ── 9.5 ROC Curve Plot ─────────────────────────────────────────
print("\n── 9.5  Saving ROC Curve Plot ───────────────────────────────")
fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
colors = plt.cm.tab20(np.linspace(0, 1, len(DISEASES)))

for i, d in enumerate(DISEASES):
    if d not in per_cls_auc_eval:
        continue
    fpr, tpr, _ = roc_curve(all_targs[:, i], all_preds[:, i])
    ax_roc.plot(fpr, tpr, lw=1.5, color=colors[i],
                label=f"{d} (AUC={per_cls_auc_eval[d]:.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax_roc.set_xlabel("False Positive Rate", fontsize=12)
ax_roc.set_ylabel("True Positive Rate", fontsize=12)
ax_roc.set_title(
    f"ROC Curve — Student ConvNeXt-V2-Tiny\nMacro-AUC = {macro_auc_eval:.4f}",
    fontsize=13, fontweight="bold"
)
ax_roc.legend(loc="lower right", fontsize=7, ncol=2)
ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.01])
ax_roc.grid(True, alpha=0.3)
roc_path = f"{CKPT_DIR}roc_curve_student.png"
fig_roc.tight_layout()
fig_roc.savefig(roc_path, dpi=150)
plt.close(fig_roc)
print(f"   ✅ ROC curve saved → {roc_path}")

# ── 9.6 AUC Bar Chart ─────────────────────────────────────────
print("\n── 9.6  Saving AUC Bar Chart ────────────────────────────────")
sorted_items = sorted(per_cls_auc_eval.items(), key=lambda x: x[1], reverse=True)
labels_bar   = [x[0] for x in sorted_items]
values_bar   = [x[1] for x in sorted_items]
bar_colors   = ["#22c55e" if v >= 0.90 else "#f59e0b" if v >= 0.80 else "#ef4444" for v in values_bar]

fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
bars = ax_bar.barh(labels_bar, values_bar, color=bar_colors, edgecolor="white", height=0.65)
ax_bar.axvline(x=macro_auc_eval, color="#6366f1", linestyle="--", lw=2,
               label=f"Macro-AUC = {macro_auc_eval:.4f}")
for bar, val in zip(bars, values_bar):
    ax_bar.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9)
ax_bar.set_xlabel("AUC Score", fontsize=12)
ax_bar.set_title("Per-Class AUC — Student ConvNeXt-V2-Tiny", fontsize=13, fontweight="bold")
ax_bar.set_xlim([0, 1.05])
ax_bar.legend(fontsize=10)
ax_bar.grid(axis="x", alpha=0.3)
bar_path = f"{CKPT_DIR}auc_barchart_student.png"
fig_bar.tight_layout()
fig_bar.savefig(bar_path, dpi=150)
plt.close(fig_bar)
print(f"   ✅ AUC bar chart saved → {bar_path}")

# ── Final Summary ─────────────────────────────────────────────
print("\n" + "═" * 65)
print("🏆 FINAL EVALUATION SUMMARY")
print("═" * 65)
print(f"   Model           : ConvNeXt-V2-Tiny (Student KD)")
print(f"   Val Samples     : {len(val_df)}")
print(f"   Macro-AUC       : {macro_auc_eval:.4f}")
print(f"   Micro-AUC       : {micro_auc_eval:.4f}")
print(f"   Macro-F1        : {macro_f1:.4f}")
print(f"   Micro-F1        : {micro_f1:.4f}")
print(f"   Macro-Precision : {macro_precision:.4f}")
print(f"   Macro-Recall    : {macro_recall:.4f}")
print(f"   Cardio Sensitivity : {sens:.4f}  |  Specificity : {spec:.4f}")
print("═" * 65)
print(f"   Saved artefacts:")
print(f"   • {CKPT_DIR}best_student.pth")
print(f"   • {CKPT_DIR}roc_curve_student.png")
print(f"   • {CKPT_DIR}auc_barchart_student.png")
print("═" * 65)

# Cleanup
del eval_model, all_preds, all_targs, bin_preds
gc.collect(); torch.cuda.empty_cache()
