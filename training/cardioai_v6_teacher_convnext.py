# ================================================================
# CARDIOAI V6: CONVNEXT-BASE TEACHER TRAINING PIPELINE
# Architecture : ConvNeXt-V2 Base (ImageNet-22k pretrained)
# Goal         : Push Macro-AUC → 0.88+ on NIH-14
# Image Size   : 384×384 (matches ConvNeXt-V2 pretraining resolution)
#                Kaggle T4 (16GB): BATCH=8, ACCUM=4 → effective 32
# Key Changes  :
#   1. Pre-computed .npy CLAHE cache (zero I/O bottleneck during training)
#   2. 384×384 input (optimal for ConvNeXt-V2 VRAM vs AUC trade-off)
#   3. Medical-safe augmentation + 20% MixUp after warmup
#   4. ImageNet normalisation (correct for ConvNeXt)
#   5. Label-smoothing on hard labels
#   6. Class-imbalance: AsymmetricLoss + per-class pos-weights
#   7. Warm-up + cosine-decay LR; hard-mining inside loss
#   8. CPU-memory freed after every epoch (gc + cache flush)
#   9. No pipeline bottleneck: persistent workers, pin_memory
# ================================================================

import os
# ── PIP INSTALL FOR KAGGLE/COLAB COMPATIBILITY ────────────────────
# ⚠️ RUN THIS IN A SEPARATE JUPYTER CELL INSTEAD OF INSIDE THE SCRIPT:
# !pip install -q timm opencv-python-headless huggingface_hub
# ─────────────────────────────────────────────────────────────────
import gc, glob, cv2, time, shutil
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

# ── 00. REPRODUCIBILITY SEED ──────────────────────────────────────
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

# ── 0. HUGGING FACE TOKEN (KAGGLE SECRETS) ───────────────────────
# Needed if downloading gated weights, or to prevent rate limits.
try:
    from kaggle_secrets import UserSecretsClient
    import huggingface_hub
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    huggingface_hub.login(token=hf_token)
    os.environ["HF_TOKEN"] = hf_token
    print("✅ Hugging Face token loaded successfully from Kaggle Secrets.")
except ImportError:
    print("⚠️  Not on Kaggle (or kaggle_secrets missing). Using public weights.")
except Exception:
    print("⚠️  No HF_TOKEN found in Kaggle Secrets. Using public weights.")
# ─────────────────────────────────────────────────────────────────

# ── 1. SANITY CHECK: opencv-python must be installed ──────────────
# pip install opencv-python-headless  (Kaggle already has it)
# ─────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════════
DATA_DIR   = "/kaggle/input/datasets/organizations/nih-chest-xrays/data/"
META_PATH  = DATA_DIR + "Data_Entry_2017.csv"
CKPT_DIR   = "/kaggle/working/"

num_gpus   = 1 # 🚀 Forced to 1: Profiling shows DP sync is SLOWER than single GPU
IMG_SIZE = 384    # Training resolution (constant to avoid performance drops)

# BATCH_SIZE: 8 prevents CUDA Out Of Memory (OOM) on massive 512px inputs
BATCH_SIZE  = 8 * num_gpus  

# ── CACHE SETTINGS ────────────────────────────────────────────────
USE_PRECOMPUTED_CACHE = True

if USE_PRECOMPUTED_CACHE:
    # Clean, direct pointers to your new fixed cache dataset!
    # (Checking multiple paths to cover both Kagglehub downloads & Kaggle UI Attachments)
    path1 = "/kaggle/input/nih-cache-384-fixed/my_cache/"
    path2 = "/kaggle/input/datasets/mahantsuman/nih-cache-384-fixed/my_cache/"
    
    if os.path.exists(path1):
        CACHE_DIR = path1
    elif os.path.exists(path2):
        CACHE_DIR = path2
    else:
        # Fallback automated exact-download via KaggleHub
        import kagglehub
        dl_path = kagglehub.dataset_download("mahantsuman/nih-cache-384-fixed")
        CACHE_DIR = os.path.join(dl_path, "my_cache/")
        
    print(f"📦 Precomputed Cache Loaded Successfully from: {CACHE_DIR}")

else:
    CACHE_DIR = "/kaggle/working/img_cache_npy/"         # Build it dynamically

EPOCHS      = 14
WARMUP_EP   = 2
BASE_LR     = 5e-5
MIN_LR      = 1e-6
ACCUM_STEPS = 4
EMA_DECAY   = 0.9998

EARLY_STOP_PATIENCE = 5
epochs_no_improve   = 0

RESUME_EPOCH    = 5
RESUME_BEST_AUC = 0.8483

# Label smoothing: convert 0→ε, 1→1-ε (reduces overconfidence)
LABEL_SMOOTH_EPS = 0.05  # 5 % smoothing is safe for multi-label

# 4 workers = Kaggle's CPU count. More causes the DataLoader warning we already saw.
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prevent OpenCV from spawning its own thread pool inside each worker.
# Without this: 4 workers × N OpenCV threads = thread contention = slowdown.
cv2.setNumThreads(0)

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(DISEASES)

torch.backends.cudnn.benchmark = True
# TF32 is enabled for architectures that support it (Ampere+). 
# Note: Kaggle uses T4 GPUs (Turing), so TF32 will silently fall back to FP32.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("=" * 65)
print(f"🚀 CARDIOAI V6 TEACHER  ▸  ConvNeXt-Base  ▸  NIH-14")
print(f"   HARDWARE : {DEVICE}  |  GPUs: {num_gpus}")
print(f"   IMG SIZE : {IMG_SIZE}   BATCH: {BATCH_SIZE}")
print(f"   EPOCHS   : {EPOCHS}  (warm-up: {WARMUP_EP})")
print("=" * 65)


# ════════════════════════════════════════════════════════════════
# 2. DATASET  ─  with CLAHE, Medical-Safe Augmentation & Label Smoothing
# ════════════════════════════════════════════════════════════════

# ── 2a. CLAHE  ─  object created ONCE at module load, reused every image ──────
# KEY FIX: cv2.createCLAHE() was called inside __getitem__, allocating a new
# C++ object for every single image (86,000+/epoch → massive CPU overhead).
# Creating it here once and sharing via fork-copy is identical CLAHE math,
# just without the repeated allocation cost. Zero AUC impact.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """
    CLAHE: Enhances local contrast in lung fields without amplifying noise.
    clip_limit=2.0, tileGridSize=(8,8) are the medical-imaging gold standards.
    """
    gray = np.array(pil_img.convert("L"))   # uint8 H×W
    eq   = _CLAHE.apply(gray)               # ← reuses the cached object
    rgb  = np.stack([eq, eq, eq], axis=-1)  # H×W×3 for ConvNeXt
    return Image.fromarray(rgb)             # no 'mode' arg → no Pillow 13 warning


# ── 2b. Label smoothing helper ───────────────────────────────────
def smooth_labels(hard: np.ndarray, eps: float = LABEL_SMOOTH_EPS) -> np.ndarray:
    """
    Converts {0,1} hard labels to {ε, 1-ε} soft labels.
    Prevents the model from being maximally overconfident.
    Formula: y_soft = y_hard × (1 − ε) + (1 − y_hard) × ε
    """
    return hard * (1.0 - eps) + (1.0 - hard) * eps


# ── 2c. Read metadata ────────────────────────────────────────────
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

# Patient-level split (prevents data leakage across train/val)
train_pat, temp_pat = train_test_split(
    metadata["Patient ID"].unique(), test_size=0.20, random_state=42
)
val_pat, _ = train_test_split(temp_pat, test_size=0.50, random_state=42)

train_df = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)
val_df   = metadata[metadata["Patient ID"].isin(val_pat)].reset_index(drop=True)

# 🚀 100% data: ConvNeXt is well-regularized with drop_path+ASL — use all of it.
# train_df is already patient-split so there is zero data leakage.

# ════════════════════════════════════════════════════════════════
# 2d. ONE-TIME OFFLINE PREPROCESSING: CLAHE + Resize → disk cache
#     Runs ONCE before training. Every epoch just reads tiny 320px PNGs.
#     Eliminates ALL CLAHE / resize / full-PNG-decode from the hot path.
# ════════════════════════════════════════════════════════════════
def preprocess_to_cache(df: pd.DataFrame, img_size: int, cache_dir: str):
    """
    Apply CLAHE + resize to every image ONCE and write to cache_dir.
    Training loop then just does imread(small_png) + to_tensor — no heavy CPU ops.
    Uses multi-threaded I/O for fast parallel processing.
    """
    from concurrent.futures import ThreadPoolExecutor

    if USE_PRECOMPUTED_CACHE:
        print(f"✅ Using pre-computed Kaggle Dataset at {cache_dir}. Skipping caching.")
        return

    os.makedirs(cache_dir, exist_ok=True)
    all_paths = df["image_path"].tolist()

    # Check if cache is already complete (skip if re-running the notebook)
    # FIX: Must check for .npy since we convert .png to .npy
    done = sum(1 for p in all_paths
               if os.path.exists(os.path.join(cache_dir, os.path.basename(p).replace(".png", ".npy"))))
    if done == len(all_paths):
        print(f"\u2705 Cache already complete ({done} images). Skipping preprocessing.")
        return

    print(f"\n🔄 Building image cache: CLAHE({img_size}×{img_size}) → {cache_dir}")
    print(f"   {len(all_paths) - done} images to process (out of {len(all_paths)} total)")
    print(f"   This runs ONCE. All {EPOCHS} epochs read from cache.")

    def process_one(src_path: str):
        # Swap .png for .npy to store the raw array
        dst_path = os.path.join(cache_dir, os.path.basename(src_path).replace(".png", ".npy"))
        if os.path.exists(dst_path):
            return
        gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            gray = _CLAHE.apply(gray)
        np.save(dst_path, gray)  # saves as binary uint8 array (zero decode time!)

    n_threads = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(tqdm(ex.map(process_one, all_paths),
                  total=len(all_paths), desc="  💾 Caching (.npy)"))

    cached = sum(1 for p in all_paths
                 if os.path.exists(os.path.join(cache_dir, os.path.basename(p).replace(".png", ".npy"))))
    print(f"✅ Cache complete: {cached}/{len(all_paths)} arrays saved to {cache_dir}")


# Build cache for all images (train + val) at target resolution
all_df = pd.concat([train_df, val_df], ignore_index=True)
preprocess_to_cache(all_df, IMG_SIZE, CACHE_DIR)
del all_df  # free RAM

# ── 2e. PyTorch Dataset — loads from pre-processed cache ────────
class ChestXrayDataset(Dataset):
    """
    Ultra-fast __getitem__: imread(cached 320px PNG) → stack → tensor → normalize → augment
    CLAHE and resize are done ONCE at cache-build time. Never inside the training loop.
    """
    def __init__(self, df: pd.DataFrame, augment: bool = False,
                 img_size: int = IMG_SIZE, cache_dir: str = CACHE_DIR):
        self.augment   = augment
        self.img_size  = img_size
        
        # Smart cache directory mapping (optimized for 112k files)
        self.cache_dir = cache_dir
        if os.path.exists(cache_dir):
            # Check the root directory quickly (first 50 files) to bypass deep scans
            try:
                sample_files = [f.name for f in os.scandir(cache_dir)][:50]
                if not any(f.endswith('.npy') for f in sample_files):
                    # If not in root, try one subfolder deep
                    for folder in os.scandir(cache_dir):
                        if folder.is_dir():
                            self.cache_dir = folder.path
                            break
            except Exception:
                pass

        # O(1) access: plain list, pointing to .npy files instead of .png
        self.fnames = [os.path.basename(p).replace(".png", ".npy") for p in df["image_path"].tolist()]

        # Pre-compute smoothed labels [N, 14] — no per-sample math
        raw         = np.stack(df["hard_labels"].values).astype(np.float32)
        # BUGFIX: Label smoothing (eps=0.05) breaks AsymmetricLoss + pos_weights. Disabled.
        eps         = 0.0
        self.labels = raw * (1.0 - eps) + (1.0 - raw) * eps

        # ImageNet normalization constants (allocated once)
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
        self._std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)

        # Medical-safe augmentation: lighter to prevent over-regularization
        self.aug = T.Compose([
            T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03),
                           interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.RandomHorizontalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # ① Load pre-processed numpy. Full array loaded to avoid mmap + augmentation disk seek overhead.
        path = os.path.join(self.cache_dir, self.fnames[idx])
        gray = np.load(path)

        # ② Resize if current phase requires a different resolution than cached
        if gray.shape[0] != self.img_size:
            gray = cv2.resize(gray, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR)

        # ③ Convert to 3-channel → tensor completely natively in PyTorch (Zero np.stack / permute overhead)
        img = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)  # [3,H,W]
        img.sub_(self._mean).div_(self._std)

        # ④ Medical-safe augmentation (train only)
        if self.augment:
            img = self.aug(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


# ── CLASS-BALANCED SAMPLING ─────────────────────────────────────
# Using pos_weight + AsymmetricLoss provides sufficient class imbalance handling.
# A WeightedRandomSampler on top of that runs the risk of massive over-correction.
NIH_POS_FREQ = torch.tensor([
    0.103, 0.025, 0.118, 0.177,  0.051, 0.056, 0.013, 0.047,  
    0.041, 0.021, 0.023, 0.015,  0.030, 0.002
], dtype=torch.float32)

train_loader = DataLoader(
    ChestXrayDataset(train_df, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True
)
val_loader = DataLoader(
    ChestXrayDataset(val_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True
)
print(f"✅ DataLoaders ready | train: {len(train_loader)} batches | val: {len(val_loader)} batches")



# ════════════════════════════════════════════════════════════════
# 3. CLASS-IMBALANCE  ─  AsymmetricLoss + Per-Class Pos-Weights
# ════════════════════════════════════════════════════════════════

# pos_weight = (1 - freq) / freq  → tells BCE to up-weight rare positives
NIH_POS_WEIGHTS = ((1.0 - NIH_POS_FREQ) / NIH_POS_FREQ).clamp(max=10.0).to(DEVICE)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label imbalanced classification.
    Paper: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)

    Key idea:
      - Positive samples: γ+ = 0  → no down-weighting (preserve rare signal)
      - Negative samples: γ- = 4  → hard down-weighting of easy negatives
      - Probability shifting (m): clips near-zero neg probs to reduce noisy label impact

    Combined with pos_weight from BCE to handle both:
      (a) class-level imbalance  (pos_weight)
      (b) easy-negative dominance (ASL γ-)
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma_neg  = gamma_neg
        self.gamma_pos  = gamma_pos
        self.clip       = clip
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Probabilities
        prob   = torch.sigmoid(logits)
        prob_m = torch.clamp(prob - self.clip, min=0.0, max=1.0)   # shifted neg probs

        # Asymmetric cross-entropy
        xs_pos = targets       * torch.log(prob   + 1e-8)
        xs_neg = (1 - targets) * torch.log(1 - prob_m + 1e-8)

        # Focal weights
        pt_pos = prob
        pt_neg = 1.0 - prob_m
        loss   = -(xs_pos * (1 - pt_pos) ** self.gamma_pos
                 + xs_neg * (1 - pt_neg) ** self.gamma_neg)

        # 🎯 Cleanly apply class pos-weights to positive terms
        if self.pos_weight is not None:
            loss = loss * (targets * self.pos_weight + (1 - targets))

        return loss.mean()


criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05,
                           pos_weight=NIH_POS_WEIGHTS)


# ════════════════════════════════════════════════════════════════
# 4. MODEL  ─  ConvNeXt-Base
# ════════════════════════════════════════════════════════════════
class TeacherModel(nn.Module):
    """
    ConvNeXt-V2 Base: Upgraded from V1 for much better accuracy and faster training.
    Why ConvNeXt-V2?
      • FCMAE Pretraining: Masked Auto-Encoder pretraining forces the model to 
        understand structure better than standard ImageNet classification.
      • GRN (Global Response Normalization): Added to layers to prevent 
        feature-collapse—a common issue in medical imaging with large backbones.
    """
    def __init__(self):
        super().__init__()
        self.net = timm.create_model(
            "convnextv2_base",
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_path_rate=0.3
        )

    def forward(self, x):
        return self.net(x)

model = TeacherModel().to(DEVICE)

print("⚡ Using single GPU for faster training (DataParallel overhead bypassed)")

print(f"✅ ConvNeXt-V2 Base loaded | params: "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# channels_last: [N,H,W,C] memory layout — ConvNeXt depthwise convs run 10-25% faster.
if DEVICE.type == "cuda":
    model = model.to(memory_format=torch.channels_last)
    print("✅ channels_last enabled — ConvNeXt depthwise ops ~15% faster")

# 🔥 Setup EMA Model initialized exactly like the base model
import copy
ema_model = copy.deepcopy(model)
ema_model.eval()  # EMA model is ALWAYS in eval mode
for param in ema_model.parameters():
    param.requires_grad = False
print(f"✅ EMA Model (decay {EMA_DECAY}) initialized")





# ════════════════════════════════════════════════════════════════
# 5. TRAINING STRATEGY  ─  Warm-Up + Cosine Decay + Differential LR
# ════════════════════════════════════════════════════════════════
# 🔥 Precision Layer-wise Learning Rates (Based on Medical Fine-Tuning Best Practices): 
#   Stem (Early layers): 1e-5 (0.1x of base) - preserve pretrained edge detectors
#   Stages (Middle layers): 5e-5 (0.5x of base) - adapt high level features smoothly
#   Head (Final layers): 1e-3 (10.0x of base!) - 🔥 CRITICAL for Epoch 1:
#       The head is randomly initialized. It MUST learn 10x faster than the 
#       pretrained backbone, otherwise the backbone receives garbage gradients
#       for the entire first epoch, causing AUC to stagnate at ~0.78-0.80.
optimizer = optim.AdamW([
    {"params": model.net.stem.parameters(),   "lr_mult": 0.1},
    {"params": model.net.stages.parameters(), "lr_mult": 0.5},
    {"params": model.net.head.parameters(),   "lr_mult": 10.0},
], lr=MIN_LR, weight_decay=0.1)

scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

def get_lr(epoch: int) -> float:
    if epoch < WARMUP_EP:
        return MIN_LR + (BASE_LR - MIN_LR) * (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * progress))

def set_lr(optimizer, base_lr: float):
    # DYNAMIC SCALING: multiplies the curve's current base_lr by the layer's multiplier
    for pg in optimizer.param_groups:
        mult = pg.get("lr_mult", 1.0)
        pg["lr"] = base_lr * mult


# ════════════════════════════════════════════════════════════════
# 6. AUTO-RESUME
# ════════════════════════════════════════════════════════════════
START_EPOCH = RESUME_EPOCH
best_auc    = RESUME_BEST_AUC

if START_EPOCH > 0:
    # Automatically scan for your uploaded 'myepoch' dataset
    paths_to_try = [
        "/kaggle/working/last.pth",                              # Check the active working dir FIRST
        "/kaggle/input/myepoch/last.pth",                        # Fallback to your uploaded dataset
        "/kaggle/input/datasets/symanmahant/myepoch/last.pth",
        "/kaggle/input/myepoch/myepoch/last.pth"
    ]
    
    resume_path = "/kaggle/working/last.pth" # Fallback
    for p in paths_to_try:
        if os.path.exists(p):
            resume_path = p
            break
            
    if os.path.exists(resume_path):
        ckpt      = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
        else:
            ema_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for pg in optimizer.param_groups:
            pg["weight_decay"] = 0.1
        best_auc  = ckpt.get("best_auc", RESUME_BEST_AUC) if RESUME_BEST_AUC == 0.0 else RESUME_BEST_AUC
        print(f"✅ Resumed from Epoch {START_EPOCH}  (best AUC so far: {best_auc:.4f})")
    else:
        print(f"⚠️  Checkpoint {resume_path} not found – starting fresh.")


# ════════════════════════════════════════════════════════════════
# 7. MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════
for epoch in range(START_EPOCH, EPOCHS):

    # ── 7a. Set LR (warm-up / cosine) ───────────────────────────
    current_lr = get_lr(epoch)

    for pg, orig_mult in zip(optimizer.param_groups, [0.1, 0.5, 10.0]):
        if epoch < WARMUP_EP:
            pg["lr_mult"] = orig_mult
        else:
            pg["lr_mult"] = orig_mult if orig_mult != 10.0 else 1.0

    set_lr(optimizer, current_lr)
    lrs = [f"{pg['lr']:.1e}" for pg in optimizer.param_groups]
    print(f"\nEpoch {epoch+1}/{EPOCHS}  |  BASE_LR = {current_lr:.2e}  |  Group LRs: {lrs}")

    # ── 7d. Training ─────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"  [Train]", leave=False)
    
    optimizer.zero_grad(set_to_none=True)

    for i, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)

        use_mixup = (epoch < 10) and (np.random.random() > 0.8)
        if use_mixup:
            lam = float(np.random.beta(0.4, 0.4))
            idx = torch.randperm(images.size(0), device=DEVICE)
            images = lam * images + (1 - lam) * images[idx]
            labels = lam * labels + (1 - lam) * labels[idx]

        # Forward + loss inside single autocast block
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)
            loss   = loss / ACCUM_STEPS  # 🔥 Gradient Accumulation

        scaler.scale(loss).backward()

        # 🔥 Optimizer step only after ACCUM_STEPS
        if ((i + 1) % ACCUM_STEPS == 0) or ((i + 1) == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Tightened from 1.0 to catch spikes early
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # 🔥 Update EMA Model
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(model_p.data, alpha=1.0 - EMA_DECAY)

        epoch_loss += loss.item() * ACCUM_STEPS
        pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}", lr=f"{current_lr:.1e}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"  Train loss: {avg_loss:.4f}")

    # ── 7e. Validation (Using EMA + Validation TTA) ──────────────
    inf_model = ema_model
    inf_model.eval()
    val_preds, val_targs = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  [Val]  ", leave=False):
            images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)
            h, w   = images.shape[2], images.shape[3]

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                p1 = torch.sigmoid(inf_model(images))
                p2 = torch.sigmoid(inf_model(torch.flip(images, dims=[3])))
                ch, cw = int(h * 0.9), int(w * 0.9)
                sh_, sw_ = (h - ch) // 2, (w - cw) // 2
                crop3 = F.interpolate(images[:, :, sh_:sh_+ch, sw_:sw_+cw],
                                      size=(h, w), mode="bilinear", align_corners=False)
                p3    = torch.sigmoid(inf_model(crop3))
                preds = (p1 + p2 + p3) / 3.0

            val_preds.append(preds)
            val_targs.append(labels)

    # Convert to CPU once at the end instead of every batch (CPU bottleneck fix)
    val_preds = torch.cat(val_preds).cpu().numpy()
    val_targs = torch.cat(val_targs).cpu().numpy()

    try:
        # Per-class AUC for diagnostics
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

    if macro_auc > best_auc:
        best_auc          = macro_auc
        epochs_no_improve = 0
        core_state = ema_model.module.state_dict() if hasattr(ema_model, "module") else ema_model.state_dict()
        torch.save(core_state, f"{CKPT_DIR}best_convnext_teacher.pth")
        print(f"  NEW BEST AUC: {best_auc:.4f} -> best_convnext_teacher.pth")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs.")

    torch.save({
        "epoch"    : epoch + 1,
        "model"    : model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_auc" : best_auc,
        "auc"      : macro_auc,
    }, f"{CKPT_DIR}last.pth")
    
    # ── 7f. FREE CPU MEMORY AFTER EVERY EPOCH ───────────────────
    # We leave val_preds alive so Early Stopping doesn't crash the threshold tuner.
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  🧹 CPU/GPU cache cleared (Epoch {epoch+1} done)")
    
    # ── 7g. THREAD-SAFE COOLDOWN ────────────────────────────────
    # Pause for 5 seconds to ensure background DataLoader workers and the OS
    # fully complete garbage collection before starting the next intensive epoch.
    time.sleep(5)
    
    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n  Early stopping: No improvement for {EARLY_STOP_PATIENCE} epochs. Best AUC: {best_auc:.4f}")
        break

print("=" * 65)

# ── 8a. THRESHOLD TUNING (F1 OPTIMIZATION) ─────────────────────
from sklearn.metrics import f1_score
if 'val_preds' in locals() and 'val_targs' in locals():
    print("\n🔍 Tuning Optimal Thresholds per Class...")
    best_thresh = []

    for i in range(NUM_CLASSES):
        best = 0
        best_t = 0.5
        for t in np.linspace(0.05, 0.95, 100):  # Finer resolution: 100 steps vs 50
            try:
                score = f1_score(val_targs[:, i], val_preds[:, i] > t, zero_division=0)
                if score > best:
                    best = score
                    best_t = t
            except Exception:
                pass
        best_thresh.append(best_t)

    print("🎯 Best Thresholds per Class:", {k: round(v, 2) for k, v in zip(DISEASES, best_thresh)})
else:
    print("\n⚠️ Training loop skipped. Skipping threshold tuning.")

# ── 8a. KEEP CACHE FOR REUSE ────────────────────────────────────
# Note: In Kaggle, you CANNOT programmatically move files to `/kaggle/input/` 
# because it is a strict **read-only** directory mounted by Google.
# However, because we fixed the Checkpoint bug (we only save 2 files now),
# the 12GB cache + 1GB checkpoints = 13GB. You have 7GB of free disk space remaining!
# Pseudo-labeling only outputs a tiny 5MB CSV file, so you will NOT run out of disk space.
# We will gracefully LEAVE the cache here so you don't have to rebuild it if it crashes.
if os.path.exists(CACHE_DIR) and "working" in CACHE_DIR:
    print(f"\n📁 Leaving heavy .npy cache in {CACHE_DIR} so you can reuse it later!")
    # shutil.rmtree(CACHE_DIR)  <-- DISABLED so you don't lose the cache
    pass

# ════════════════════════════════════════════════════════════════
# 9. PSEUDO-LABELING FOR KNOWLEDGE DISTILLATION (STUDENT TRAINING)
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("🎯 STARTING PSEUDO-LABELING FOR DISTILLATION")
print("   Generating soft-labels for the entire training set (100%)...")
print("=" * 65)

# 1. Recreate the full training dataframe (since we only used 40% for training)
train_df_full = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)

class PseudoLabelDataset(Dataset):
    """
    Fast pseudo-label dataset: reads from pre-built .npy cache (same as training).
    Falls back to raw PNG decoding only if a sample is not found in the cache.
    """
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
            # Fast path: read pre-computed CLAHE .npy (same format as training)
            gray = np.load(cache_path)
        else:
            # Fallback: live decode from raw PNG
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

# 2. Load the BEST weights into EMA model for cleanest pseudo-labels
# ⚠️  CRITICAL: best_convnext_teacher.pth stores EMA weights (not the raw model).
#     Load them into ema_model, NOT model, so inference uses the smoothed weights.

best_try_paths = [
    "/kaggle/working/best_convnext_teacher.pth",               # If it beats 0.8483, it saves here
    "/kaggle/input/myepoch/best_convnext_teacher.pth",         # Fallback to your uploaded best
    "/kaggle/input/datasets/symanmahant/myepoch/best_convnext_teacher.pth",
    "/kaggle/input/myepoch/myepoch/best_convnext_teacher.pth"
]

best_path = None
for p in best_try_paths:
    if os.path.exists(p):
        best_path = p
        break

try:
    if best_path:
        ema_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"✅ Loaded {best_path} for pseudo-labeling.")
    else:
        raise FileNotFoundError("Best weight file not found")
except Exception as e:
    print(f"⚠️ Could not load best teacher weights: {e}. Using current ema_model weights.")

ema_model.eval()
all_soft_labels = []

# 3. Infer Soft Labels
print(f"⏳ Inferring soft-labels (w/ TTA enabled) for {len(train_df_full)} images directly from PNGs...")
with torch.no_grad():
    for images in tqdm(pseudo_loader, desc="  [Pseudo-Labeling]"):
        images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            # 🔥 CRITICAL: Use EMA model (not raw model) — EMA gives smoother probabilities
            # that are better calibrated for downstream Student Distillation
            probs1 = torch.sigmoid(ema_model(images))
            probs2 = torch.sigmoid(ema_model(torch.flip(images, dims=[3])))
            probs  = (probs1 + probs2) / 2.0
            
        all_soft_labels.append(probs.cpu())

all_soft_labels = torch.cat(all_soft_labels).numpy()

# ─────────────────────────────────────────────────────────────────
# 📦 RAW NPY EXPORT (FOR NOTEBOOK C ENSEMBLE FUSION)
# ─────────────────────────────────────────────────────────────────
np.save(f"{CKPT_DIR}convnext_preds.npy", all_soft_labels)
print(f"📦 Saved raw probabilities direct to {CKPT_DIR}convnext_preds.npy")

# 🔥 LEVEL 4: PER-CLASS THRESHOLD FILTERING
high_thresh = np.full(NUM_CLASSES, 0.7)
low_thresh  = np.full(NUM_CLASSES, 0.3)

custom_thresh = {
    "Hernia":       (0.2, 0.05),
    "Cardiomegaly": (0.5, 0.2),
    "Effusion":     (0.6, 0.2)
}
for i, d in enumerate(DISEASES):
    if d in custom_thresh:
        high_thresh[i] = custom_thresh[d][0]
        low_thresh[i]  = custom_thresh[d][1]

# Mask of confident predictions (🔥 LEVEL 2: CONFIDENCE FILTER)
mask = (all_soft_labels > high_thresh) | (all_soft_labels < low_thresh)

# 🔥 LEVEL 3: MIX HARD + SOFT LABELS (50/50 balanced blend)
# 50% ground truth prevents fully relying on noisy teacher predictions
# 50% soft labels transfer calibrated uncertainty knowledge to student
hard_labels  = np.stack(train_df_full["hard_labels"].values).astype(np.float32)
final_labels = 0.5 * hard_labels + 0.5 * all_soft_labels

# Keep confident predictions, mark uncertain as -1 (Ignored during Student Training)
final_labels[~mask] = -1.0

# 4. Save to CSV
soft_cols = [f"soft_{d}" for d in DISEASES]
soft_df   = pd.DataFrame(final_labels, columns=soft_cols)

# Concatenate with original metadata
pseudo_df = pd.concat([
    train_df_full[["Image Index", "Patient ID", "Finding Labels"]], 
    soft_df
], axis=1)

pseudo_csv_path = f"{CKPT_DIR}nih_pseudo_labels.csv"
pseudo_df.to_csv(pseudo_csv_path, index=False)

print(f"🔥 Pseudo-labeling complete! Saved to: {pseudo_csv_path}")
print("   Upload this CSV alongside your best weights for Student Knowledge Distillation!")
print("=" * 65 + "\n")
