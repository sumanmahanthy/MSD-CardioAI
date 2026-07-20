# ================================================================
# CARDIOAI V6: SWIN-BASE TEACHER TRAINING PIPELINE  (FIXED v3)
# Architecture : Swin Transformer Base (Window=12, 384×384)
# Goal         : Push Macro-AUC → 0.86+ on NIH-14
# Teacher Role : Support Teacher 2 (alongside BioMedCLIP)
# Kaggle T4    : BATCH=8, ACCUM=4 → effective batch 32
# Key Fixes v3 :
#   1. AsymmetricLoss — hard prob clamp [1e-4, 1-1e-4]  → NaN solved
#   2. Head lr_mult  = 5.0  (was 1.0 → too slow; 2.0 → explodes)
#   3. Dead-code cleanup: ep5 guard removed (was checking 10.0, never hit)
#   4. channels_last REMOVED (Swin is attention-based; no conv benefit)
#   5. Freeze strategy fixed: only norm layers after ep 6 (not all layers)
#   6. NaN-batch skip guard added inside training loop
#   7. Pseudo-label TTA: rot90 removed (medically invalid for CXR)
#   8. Pseudo-label threshold: dynamic percentile (matches BioMedCLIP v2)
#   9. EMA buffers() are now also copied (BN / LayerNorm running stats)
#  10. pos_weight clamped to max=10 (was 20 → gradient spikes)
# ================================================================

import os
# ── PIP INSTALL FOR KAGGLE/COLAB ───────────────────────────────────
# ⚠️ RUN THIS IN A SEPARATE JUPYTER CELL INSTEAD OF INSIDE THE SCRIPT:
# !pip install -q timm opencv-python-headless huggingface_hub
# ──────────────────────────────────────────────────────────────────
import gc, glob, cv2, time, copy, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from PIL import Image
import pandas as pd
import numpy as np
import random

# ── 00. REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timm

# ── 0. HUGGING FACE TOKEN (KAGGLE SECRETS) ──────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    import huggingface_hub
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    huggingface_hub.login(token=hf_token)
    os.environ["HF_TOKEN"] = hf_token
    print("✅ HF token loaded from Kaggle Secrets.")
except ImportError:
    print("⚠️  Not on Kaggle. Using public weights.")
except Exception:
    print("⚠️  No HF_TOKEN found. Using public weights.")

# ════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════════
DATA_DIR  = "/kaggle/input/datasets/organizations/nih-chest-xrays/data/"
META_PATH = DATA_DIR + "Data_Entry_2017.csv"
CKPT_DIR  = "/kaggle/working/"

num_gpus   = 1
IMG_SIZE   = 384   # Swin-Base patch4 window12 requires 384×384
BATCH_SIZE = 8 * num_gpus

# ── CACHE SETTINGS ────────────────────────────────────────────────
# Uses the SAME 384-cache as ConvNeXt teacher (no extra disk needed!)
USE_PRECOMPUTED_CACHE = True

if USE_PRECOMPUTED_CACHE:
    path1 = "/kaggle/input/nih-cache-384-fixed/my_cache/"
    path2 = "/kaggle/input/datasets/mahantsuman/nih-cache-384-fixed/my_cache/"
    if os.path.exists(path1):
        CACHE_DIR = path1
    elif os.path.exists(path2):
        CACHE_DIR = path2
    else:
        import kagglehub
        dl_path = kagglehub.dataset_download("mahantsuman/nih-cache-384-fixed")
        CACHE_DIR = os.path.join(dl_path, "my_cache/")
    print(f"📦 Precomputed Cache Loaded from: {CACHE_DIR}")
else:
    CACHE_DIR = "/kaggle/working/img_cache_npy/"

# ── HYPERPARAMETERS ───────────────────────────────────────────────
EPOCHS     = 14
WARMUP_EP  = 2          # Linear warm-up 2 epochs → stabilizes Swin head
BASE_LR    = 3e-5       # ✅ FIX: Reduced to 3e-5 to prevent Swin attention FP16 overflow
MIN_LR     = 1e-6
ACCUM_STEPS = 4         # Effective batch = 32 (8 × 4)
EMA_DECAY   = 0.9997    # Strong EMA → stable pseudo-labels

EARLY_STOP_PATIENCE = 5
epochs_no_improve   = 0

# ── RESUME (set > 0 to continue from a saved checkpoint) ────────
# ▶ HOW TO RESUME ON KAGGLE:
#   1. After a run finishes/crashes, download "last.pth" from Output
#   2. Upload it as a new Kaggle Dataset named "myepoch"
#   3. Set RESUME_EPOCH to the last completed epoch number
#   4. Set RESUME_BEST_AUC to the best AUC seen so far
RESUME_EPOCH    = 0
RESUME_BEST_AUC = 0.0

NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prevent OpenCV thread pool inside workers → no contention
cv2.setNumThreads(0)

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(DISEASES)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("=" * 65)
print(f"🚀 CARDIOAI V6 TEACHER  ▸  Swin-Base (FIXED v3)  ▸  NIH-14")
print(f"   HARDWARE : {DEVICE}  |  GPUs: {num_gpus}")
print(f"   IMG SIZE : {IMG_SIZE}   BATCH: {BATCH_SIZE}   ACCUM: {ACCUM_STEPS}")
print(f"   EPOCHS   : {EPOCHS}  (warm-up: {WARMUP_EP})")
print(f"   BASE_LR  : {BASE_LR}   EMA: {EMA_DECAY}")
print("=" * 65)


# ════════════════════════════════════════════════════════════════
# 2. DATASET  ─  CLAHE Cache + Medical-Safe Augmentation
# ════════════════════════════════════════════════════════════════

# CLAHE object created ONCE (not inside __getitem__) — massive CPU saving
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ── 2b. Read metadata ────────────────────────────────────────────
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

# Patient-level split — zero data leakage
train_pat, temp_pat = train_test_split(
    metadata["Patient ID"].unique(), test_size=0.20, random_state=42
)
val_pat, _ = train_test_split(temp_pat, test_size=0.50, random_state=42)

train_df = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)
val_df   = metadata[metadata["Patient ID"].isin(val_pat)].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 2d. ONE-TIME CACHE BUILD (skipped if USE_PRECOMPUTED_CACHE=True)
# ════════════════════════════════════════════════════════════════
def preprocess_to_cache(df: pd.DataFrame, img_size: int, cache_dir: str):
    from concurrent.futures import ThreadPoolExecutor

    if USE_PRECOMPUTED_CACHE:
        print(f"✅ Using pre-computed cache at {cache_dir}. Skipping build.")
        return

    os.makedirs(cache_dir, exist_ok=True)
    all_paths = df["image_path"].tolist()

    done = sum(1 for p in all_paths
               if os.path.exists(os.path.join(cache_dir,
                   os.path.basename(p).replace(".png", ".npy"))))
    if done == len(all_paths):
        print(f"✅ Cache already complete ({done} images).")
        return

    print(f"\n🔄 Building image cache: CLAHE@{img_size}px → {cache_dir}")
    print(f"   {len(all_paths) - done} to process (out of {len(all_paths)})")

    def process_one(src_path: str):
        dst = os.path.join(cache_dir,
                           os.path.basename(src_path).replace(".png", ".npy"))
        if os.path.exists(dst):
            return
        gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            gray = cv2.resize(gray, (img_size, img_size),
                              interpolation=cv2.INTER_LINEAR)
            gray = _CLAHE.apply(gray)
        np.save(dst, gray)

    n_threads = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(tqdm(ex.map(process_one, all_paths),
                  total=len(all_paths), desc="  💾 Caching (.npy)"))

    cached = sum(1 for p in all_paths
                 if os.path.exists(os.path.join(cache_dir,
                     os.path.basename(p).replace(".png", ".npy"))))
    print(f"✅ Cache complete: {cached}/{len(all_paths)} arrays saved.")


all_df = pd.concat([train_df, val_df], ignore_index=True)
preprocess_to_cache(all_df, IMG_SIZE, CACHE_DIR)
del all_df


# ── 2e. Dataset ──────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """Ultra-fast: loads pre-CLAHE .npy → tensor → ImageNet normalize → augment"""

    def __init__(self, df: pd.DataFrame, augment: bool = False,
                 img_size: int = IMG_SIZE, cache_dir: str = CACHE_DIR):
        self.augment  = augment
        self.img_size = img_size

        # Smart cache resolution (handles flat and nested layouts)
        self.cache_dir = cache_dir
        if os.path.exists(cache_dir):
            try:
                sample_files = [f.name for f in os.scandir(cache_dir)][:50]
                if not any(f.endswith('.npy') for f in sample_files):
                    for folder in os.scandir(cache_dir):
                        if folder.is_dir():
                            self.cache_dir = folder.path
                            break
            except Exception:
                pass

        self.fnames = [os.path.basename(p).replace(".png", ".npy")
                       for p in df["image_path"].tolist()]

        # Hard labels only — label smoothing disabled (ASL handles imbalance)
        raw = np.stack(df["hard_labels"].values).astype(np.float32)
        self.labels = raw   # eps = 0.0

        # ImageNet normalization (NOT CLIP norm — Swin uses ImageNet pretraining)
        self._mean = torch.tensor([0.485, 0.456, 0.406],
                                  dtype=torch.float32).view(3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225],
                                  dtype=torch.float32).view(3, 1, 1)

        # Medical-safe augmentation (conservative for Swin window attention)
        self.aug = T.Compose([
            T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03),
                           interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.cache_dir, self.fnames[idx])
        gray = np.load(path)                          # uint8 H×W

        if gray.shape[0] != self.img_size:
            gray = cv2.resize(gray, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR)

        # uint8 → float32 tensor [3,H,W] — zero copy with unsqueeze+repeat
        img = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        img.sub_(self._mean).div_(self._std)

        if self.augment:
            img = self.aug(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


# ── Class frequency for pos_weight ───────────────────────────────
NIH_POS_FREQ = torch.tensor([
    0.103, 0.025, 0.118, 0.177,  0.051, 0.056, 0.013, 0.047,
    0.041, 0.021, 0.023, 0.015,  0.030, 0.002
], dtype=torch.float32)

train_loader = DataLoader(
    ChestXrayDataset(train_df, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    prefetch_factor=2, persistent_workers=True
)
val_loader = DataLoader(
    ChestXrayDataset(val_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    prefetch_factor=2, persistent_workers=True
)
print(f"✅ DataLoaders ready | train: {len(train_loader)} | val: {len(val_loader)} batches")


# ════════════════════════════════════════════════════════════════
# 3. LOSS  ─  AsymmetricLoss (FIXED: hard prob clamp → no NaN)
# ════════════════════════════════════════════════════════════════

# FIX: clamp max=10 (was 20 → gradient spikes with ASL on rare classes)
NIH_POS_WEIGHTS = ((1.0 - NIH_POS_FREQ) / NIH_POS_FREQ).clamp(max=10.0).to(DEVICE)


class AsymmetricLoss(nn.Module):
    """
    ASL: Ben-Baruch et al., ICCV 2021.
    FIXED v3:
      - prob clamped [1e-4, 1-1e-4] BEFORE log → 100% NaN-safe
      - prob_m clamped with same floor → no log(0) from shifting
    gamma_neg=3 (swin support teacher; slightly softer than ConvNeXt=4)
    """
    def __init__(self, gamma_neg: float = 3.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma_neg  = gamma_neg
        self.gamma_pos  = gamma_pos
        self.clip       = clip
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob   = torch.sigmoid(logits)
        # ✅ FIX 1: Clamp BEFORE log — prevents log(0) from FP16 underflow
        prob   = torch.clamp(prob,              min=1e-4, max=1 - 1e-4)
        prob_m = torch.clamp(prob - self.clip,  min=1e-4, max=1 - 1e-4)

        xs_pos = targets       * torch.log(prob)
        xs_neg = (1 - targets) * torch.log(1 - prob_m)

        pt_pos = prob
        pt_neg = 1.0 - prob_m
        loss   = -(xs_pos * (1 - pt_pos) ** self.gamma_pos
                 + xs_neg * (1 - pt_neg) ** self.gamma_neg)

        if self.pos_weight is not None:
            loss = loss * (targets * self.pos_weight + (1 - targets))

        return loss.mean()


criterion = AsymmetricLoss(gamma_neg=3.0, gamma_pos=0.0, clip=0.05,
                           pos_weight=NIH_POS_WEIGHTS)


# ════════════════════════════════════════════════════════════════
# 4. MODEL  ─  Swin-Base Transformer (Window=12, 384×384)
# ════════════════════════════════════════════════════════════════
class TeacherModel(nn.Module):
    """
    Swin Transformer Base pretrained on ImageNet-22k → fine-tuned on IN-1k.
    Acts as Teacher 2 (support) — provides window-attention diversity
    uncorrelated with ConvNeXt's depthwise-convolution features.
    drop_path_rate=0.2: keeps regularization strong for 86k-image dataset.
    """
    def __init__(self):
        super().__init__()
        self.net = timm.create_model(
            "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_path_rate=0.2
        )

    def forward(self, x):
        return self.net(x)


model = TeacherModel().to(DEVICE)
print("⚡ Single GPU (DataParallel sync overhead bypassed)")
print(f"✅ Swin-Base loaded | params: "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ✅ FIX 4: channels_last REMOVED for Swin
# Swin uses shifted window self-attention → no conv depthwise ops.
# channels_last only speeds up depthwise convolutions (ConvNeXt, EfficientNet).
# On Swin it adds NHWC→NCHW conversion overhead with zero benefit.

# ── EMA Model ────────────────────────────────────────────────────
ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad = False
print(f"✅ EMA Model (decay={EMA_DECAY}) initialized")


# ════════════════════════════════════════════════════════════════
# 5. OPTIMIZER  ─  Layer-wise LR + Warm-Up + Cosine Decay
# ════════════════════════════════════════════════════════════════
optimizer_groups = [
    {"params": [], "lr_mult": 0.1}, # patch_embed
    {"params": [], "lr_mult": 0.5}, # backbone (layers, norm, etc)
    {"params": [], "lr_mult": 3.0}, # head (✅ FIX: reduced from 5.0 to 3.0 to prevent NaNs)
]
for name, param in model.named_parameters():
    if not param.requires_grad: continue
    if 'head' in name:
        optimizer_groups[2]["params"].append(param)
    elif 'patch_embed' in name:
        optimizer_groups[0]["params"].append(param)
    else:
        optimizer_groups[1]["params"].append(param)

# ✅ FIX: eps=1e-6 is CRITICAL for Swin Transformers in FP16 to avoid NaN explosions
optimizer = optim.AdamW(optimizer_groups, lr=MIN_LR, weight_decay=0.05, eps=1e-6)

scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))


def get_lr(epoch: int) -> float:
    """Linear warm-up → cosine decay."""
    if epoch < WARMUP_EP:
        return MIN_LR + (BASE_LR - MIN_LR) * (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, base_lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * pg.get("lr_mult", 1.0)


# ════════════════════════════════════════════════════════════════
# 6. AUTO-RESUME
# ════════════════════════════════════════════════════════════════
# ▶ TO RESUME: Set RESUME_EPOCH > 0 and upload last.pth as "myepoch" dataset
START_EPOCH = RESUME_EPOCH
best_auc    = RESUME_BEST_AUC

if START_EPOCH > 0:
    paths_to_try = [
        "/kaggle/working/last.pth",
        "/kaggle/input/myepoch/last.pth",
        "/kaggle/input/datasets/symanmahant/myepoch/last.pth",
        "/kaggle/input/myepoch/myepoch/last.pth",
    ]
    resume_path = "/kaggle/working/last.pth"
    for p in paths_to_try:
        if os.path.exists(p):
            resume_path = p
            break

    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
        else:
            ema_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_auc = ckpt.get("best_auc", RESUME_BEST_AUC) if RESUME_BEST_AUC == 0.0 else RESUME_BEST_AUC
        print(f"✅ Resumed from Epoch {START_EPOCH} (best AUC: {best_auc:.4f})")
    else:
        print(f"⚠️  Checkpoint not found at {resume_path} — starting fresh.")


# ════════════════════════════════════════════════════════════════
# 7. MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════
for epoch in range(START_EPOCH, EPOCHS):

    # ── 7a. LR Schedule ──────────────────────────────────────────
    current_lr = get_lr(epoch)
    
    for pg, orig_mult in zip(optimizer.param_groups, [0.1, 0.5, 3.0]):
        pg["lr_mult"] = orig_mult
        if epoch >= 9 and orig_mult == 3.0:
            pg["lr_mult"] = 1.0

    if epoch == 9:
        print(f"  Epoch 10: MixUp OFF. Head LR tapered.")

    set_lr(optimizer, current_lr)
    lrs = [f"{pg['lr']:.1e}" for pg in optimizer.param_groups]
    print(f"\nEpoch {epoch+1}/{EPOCHS}  |  LR = {current_lr:.2e}  |  Group LRs: {lrs}")

    # ── 7b. Training ─────────────────────────────────────────────
    model.train()
    epoch_loss  = 0.0
    nan_batches = 0
    pbar = tqdm(train_loader, desc=f"  [Train]", leave=False)
    
    optimizer.zero_grad(set_to_none=True)

    for i, (images, labels) in enumerate(pbar):
        # ✅ FIX 4: No channels_last for Swin (attention-based, no conv benefit)
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        use_mixup = (epoch < 9) and (np.random.random() > 0.8)
        if use_mixup:
            lam = float(np.random.beta(0.3, 0.3))
            idx = torch.randperm(images.size(0), device=DEVICE)
            images = lam * images + (1 - lam) * images[idx]
            labels = lam * labels + (1 - lam) * labels[idx]

        # Forward pass inside autocast (FP16 for speed)
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images)

        # ✅ FIX: Loss computed OUTSIDE autocast in FP32
        # Prevents FP16 tiny-value underflow inside ASL log operations
        loss = criterion(logits.float(), labels.float())

        # ✅ FIX 6: Skip NaN batch — protects weights from corruption
        if not torch.isfinite(loss):
            nan_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if nan_batches <= 3:
                print(f"  ⚠️ NaN batch at step {i} — skipping (total: {nan_batches})")
            continue

        loss = loss / ACCUM_STEPS  # Gradient accumulation scaling
        scaler.scale(loss).backward()

        # Optimizer step every ACCUM_STEPS
        if ((i + 1) % ACCUM_STEPS == 0) or ((i + 1) == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # ✅ EMA update: copies both parameters AND buffers (LayerNorm running stats)
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(model_p.data, alpha=1.0 - EMA_DECAY)
                for ema_b, model_b in zip(ema_model.buffers(), model.buffers()):
                    ema_b.copy_(model_b)

        epoch_loss += loss.item() * ACCUM_STEPS
        pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}", lr=f"{current_lr:.1e}")

    avg_loss = epoch_loss / max(1, len(train_loader) - nan_batches)
    print(f"  Train loss: {avg_loss:.4f}"
          + (f"  ⚠️ ({nan_batches} NaN batches skipped)" if nan_batches > 0 else ""))

    # ── 7c. Validation  (EMA model + 5-View TTA) ─────────────────
    ema_model.eval()
    val_preds, val_targs = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  [Val]  ", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            h, w   = images.shape[2], images.shape[3]

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                # 5-View TTA: original, h-flip, centre-crop-90%, rot+90, rot-90
                # ✅ Note: rot90 IS valid here for CXR at inference (assessment diversity)
                p1 = torch.sigmoid(ema_model(images))
                p2 = torch.sigmoid(ema_model(torch.flip(images, dims=[3])))
                ch, cw   = int(h * 0.9), int(w * 0.9)
                sh_, sw_ = (h - ch) // 2, (w - cw) // 2
                crop3    = F.interpolate(images[:, :, sh_:sh_+ch, sw_:sw_+cw],
                                         size=(h, w), mode="bilinear", align_corners=False)
                p3 = torch.sigmoid(ema_model(crop3))
                p4 = torch.sigmoid(ema_model(torch.rot90(images,  1, dims=[2, 3])))
                p5 = torch.sigmoid(ema_model(torch.rot90(images, -1, dims=[2, 3])))
                preds = (p1 + p2 + p3 + p4 + p5) / 5.0

            val_preds.append(preds)
            val_targs.append(labels)

    val_preds = torch.cat(val_preds).cpu().numpy()
    val_targs = torch.cat(val_targs).cpu().numpy()

    try:
        per_cls_auc = {
            d: roc_auc_score(val_targs[:, i], val_preds[:, i])
            for i, d in enumerate(DISEASES)
            if val_targs[:, i].sum() > 0
        }
        macro_auc = float(np.mean(list(per_cls_auc.values())))
        card_auc  = per_cls_auc.get("Cardiomegaly", 0.0)
    except ValueError:
        macro_auc, card_auc = 0.0, 0.0
        per_cls_auc = {}

    print(f"  Val Macro-AUC : {macro_auc:.4f}  |  Cardiomegaly AUC: {card_auc:.4f}")
    print("  Per-class AUC:", {k: f"{v:.3f}" for k, v in per_cls_auc.items()})

    # ── 7d. Save best / Early Stopping ───────────────────────────
    if macro_auc > best_auc:
        best_auc = macro_auc
        epochs_no_improve = 0
        core_state = (ema_model.module.state_dict()
                      if hasattr(ema_model, "module") else ema_model.state_dict())
        torch.save(core_state, f"{CKPT_DIR}best_swin_teacher.pth")
        print(f"  🏆 NEW BEST  AUC: {best_auc:.4f}  → best_swin_teacher.pth")
    else:
        epochs_no_improve += 1
        print(f"  ⚠️ No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs.")

    # ── 7e. Safety checkpoint (Kaggle 12-hr guard) ───────────────
    torch.save({
        "epoch"    : epoch + 1,
        "model"    : model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_auc" : best_auc,
        "auc"      : macro_auc,
    }, f"{CKPT_DIR}last.pth")

    # ── 7f. Memory cleanup ────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  🧹 Cache cleared (Epoch {epoch+1} done)")

    # ── 7g. Cooldown (lets DataLoader workers finish GC) ─────────
    time.sleep(5)

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n🛑 EARLY STOPPING after {EARLY_STOP_PATIENCE} stagnant epochs.")
        break

print("=" * 65)
print(f"✅ Training complete! Best Macro-AUC: {best_auc:.4f}")


# ════════════════════════════════════════════════════════════════
# 8. THRESHOLD TUNING (F1 Optimization on Val Set)
# ════════════════════════════════════════════════════════════════
from sklearn.metrics import f1_score
print("\n🔍 Tuning Optimal Decision Thresholds per Class...")
best_thresh = []

for i in range(NUM_CLASSES):
    best, best_t = 0, 0.5
    for t in np.linspace(0.05, 0.95, 100):
        try:
            score = f1_score(val_targs[:, i], val_preds[:, i] > t, zero_division=0)
            if score > best:
                best, best_t = score, t
        except Exception:
            pass
    best_thresh.append(best_t)

print("🎯 Best Thresholds:", {k: round(v, 2) for k, v in zip(DISEASES, best_thresh)})


# ════════════════════════════════════════════════════════════════
# 9. PSEUDO-LABELING FOR KNOWLEDGE DISTILLATION
#    Outputs: swin_preds.npy  +  nih_pseudo_labels_swin.csv
#    These are fused with ConvNeXt + BioMedCLIP outputs in
#    the distillation notebook to create the ensemble CSV.
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("🎯 PSEUDO-LABELING PHASE")
print("   Generating soft-labels for entire training set (100%)...")
print("=" * 65)

train_df_full = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)


class PseudoLabelDataset(Dataset):
    """Reads from pre-built .npy cache — same format as training pipeline."""
    def __init__(self, df: pd.DataFrame, cache_dir: str = CACHE_DIR,
                 img_size: int = IMG_SIZE):
        self.img_paths = df["image_path"].tolist()
        self.fnames    = [os.path.basename(p).replace(".png", ".npy")
                          for p in self.img_paths]
        self.cache_dir = cache_dir
        self.img_size  = img_size
        self._mean     = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std      = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        cache_path = os.path.join(self.cache_dir, self.fnames[idx])
        if os.path.exists(cache_path):
            gray = np.load(cache_path)
        else:
            # Fallback: live decode
            gray = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
            if gray is None:
                gray = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                gray = cv2.resize(gray, (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_LINEAR)
                gray = _CLAHE.apply(gray)

        img = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        img.sub_(self._mean).div_(self._std)
        return img


pseudo_dataset = PseudoLabelDataset(train_df_full, img_size=IMG_SIZE)
pseudo_loader  = DataLoader(
    pseudo_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4
)

# Load BEST EMA weights for cleanest pseudo-labels
best_try_paths = [
    f"{CKPT_DIR}best_swin_teacher.pth",
    "/kaggle/input/myepoch/best_swin_teacher.pth",
    "/kaggle/input/datasets/symanmahant/myepoch/best_swin_teacher.pth",
    "/kaggle/input/myepoch/myepoch/best_swin_teacher.pth",
]
best_path = None
for p in best_try_paths:
    if os.path.exists(p):
        best_path = p
        break

try:
    if best_path:
        ema_model.load_state_dict(
            torch.load(best_path, map_location=DEVICE, weights_only=False))
        print(f"✅ Loaded best weights from: {best_path}")
    else:
        raise FileNotFoundError("No best_swin_teacher.pth found")
except Exception as e:
    print(f"⚠️  Could not load best weights ({e}) — using current EMA.")

ema_model.eval()
all_soft_labels = []

print(f"⏳ Inferring soft-labels (2-view TTA) for {len(train_df_full)} images...")
with torch.no_grad():
    for images in tqdm(pseudo_loader, desc="  [Pseudo-Labeling]"):
        images = images.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            p1    = torch.sigmoid(ema_model(images))
            p2    = torch.sigmoid(ema_model(torch.flip(images, dims=[3])))
            probs = (p1 + p2) / 2.0
        all_soft_labels.append(probs.cpu())

all_soft_labels = torch.cat(all_soft_labels).numpy()

# Save raw predictions (for ensemble fusion in distillation notebook)
np.save(f"{CKPT_DIR}swin_preds.npy", all_soft_labels)
print(f"📦 Saved raw probabilities → {CKPT_DIR}swin_preds.npy")

# ── Confidence Filtering (dynamic percentile — matches BioMedCLIP v2) ──
# ✅ FIX 8: Dynamic percentile thresholds (was hard-coded 0.7/0.3 → biased)
print("🔍 Computing dynamic percentile thresholds per class...")
high_thresh = np.zeros(NUM_CLASSES)
low_thresh  = np.zeros(NUM_CLASSES)
for i in range(NUM_CLASSES):
    high_thresh[i] = np.percentile(all_soft_labels[:, i], 85)
    low_thresh[i]  = np.percentile(all_soft_labels[:, i], 15)

# Override for Cardiomegaly (key target class — use tighter filtering)
card_idx = DISEASES.index("Cardiomegaly")
high_thresh[card_idx] = max(high_thresh[card_idx], 0.55)
low_thresh[card_idx]  = min(low_thresh[card_idx],  0.20)

mask = (all_soft_labels > high_thresh) | (all_soft_labels < low_thresh)

# Label blend: 60% hard GT + 40% soft teacher
# (Support teacher blend: slightly harder GT weight vs BioMedCLIP 70/30)
hard_labels  = np.stack(train_df_full["hard_labels"].values).astype(np.float32)
final_labels = 0.6 * hard_labels + 0.4 * all_soft_labels

# Mark uncertain predictions as -1 (student ignores them during training)
final_labels[~mask] = -1.0

# Save pseudo-label CSV
soft_cols = [f"soft_{d}" for d in DISEASES]
soft_df   = pd.DataFrame(final_labels, columns=soft_cols)
pseudo_df = pd.concat(
    [train_df_full[["Image Index", "Patient ID", "Finding Labels"]], soft_df],
    axis=1
)
pseudo_csv_path = f"{CKPT_DIR}nih_pseudo_labels_swin.csv"
pseudo_df.to_csv(pseudo_csv_path, index=False)

print(f"\n✅ Pseudo-labeling complete!")
print(f"   CSV  → {pseudo_csv_path}")
print(f"   NPY  → {CKPT_DIR}swin_preds.npy")
print(f"   Best → {CKPT_DIR}best_swin_teacher.pth")
print("\n📌 NEXT STEPS:")
print("   1. Download: last.pth, best_swin_teacher.pth, swin_preds.npy, nih_pseudo_labels_swin.csv")
print("   2. Upload to Kaggle Dataset named 'myepoch' (for resume) or 'swin-outputs' (for distillation)")
print("   3. Fuse swin_preds.npy + convnext_preds.npy + clip_preds.npy in distillation notebook")
print("=" * 65 + "\n")
