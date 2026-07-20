# ================================================================
# CARDIOAI V6: BIOMEDCLIP TEACHER TRAINING PIPELINE  (ADVANCED v2)
# Architecture : BioMedCLIP ViT-B/16 (microsoft/BiomedCLIP-PubMedBERT_256)
# Goal         : Push Macro-AUC → 0.85+ on NIH-14
# Key Techniques:
#   1. Progressive Layer Unfreezing (ULMFiT-style, last→first every 2 ep)
#   2. SGDR Cosine Annealing with Warm Restarts (T0=4, restarts at ep 4,8,12)
#   3. 5-View Test-Time Augmentation at validation
#   4. RandomErasing augmentation (p=0.15) for ViT context learning
#   5. logit_scale clamped [1, 20] to prevent NaN loss
#   6. EMA decay 0.9995 — very slow tracking → stable pseudo-labels
#   7. Stronger weight decay 0.10 + dropout 0.4 for fully-unfrozen ViT
# ================================================================

import os, gc, glob, cv2, time, shutil, math
# ── PIP INSTALL FOR KAGGLE/COLAB COMPATIBILITY ────────────────────
# ⚠️ RUN THIS IN A SEPARATE JUPYTER CELL INSTEAD OF INSIDE THE SCRIPT:
# !pip install -q timm open_clip_torch opencv-python-headless huggingface_hub
# ─────────────────────────────────────────────────────────────────
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
import copy

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
import open_clip

# ── 0. HUGGING FACE TOKEN (KAGGLE SECRETS) ───────────────────────
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

# ════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════════
DATA_DIR  = "/kaggle/input/datasets/organizations/nih-chest-xrays/data/"
META_PATH = DATA_DIR + "Data_Entry_2017.csv"
CKPT_DIR  = "/kaggle/working/"

num_gpus  = 1
IMG_SIZE  = 224    # BiomedCLIP requires 224×224
BATCH_SIZE = 12 * num_gpus

# ── CACHE SETTINGS ─────────────────────────────────────────────────
# ❌ FIX 1: Removed CLAHE. Must set to False to rebuild a clean cache!
USE_PRECOMPUTED_CACHE = False
if USE_PRECOMPUTED_CACHE:
    path1 = "/kaggle/input/nih-cache-224-clip/nih-cache-224-clip/"
    path2 = "/kaggle/input/nih-cache-224-clip/"
    if os.path.exists(path1):
        CACHE_DIR = path1
    elif os.path.exists(path2):
        CACHE_DIR = path2
    else:
        print("⚠️ Precomputed Cache not found! Reverting to local build in working dir.")
        CACHE_DIR = "/kaggle/working/img_cache_npy/"
else:
    CACHE_DIR = "/kaggle/working/img_cache_npy/"

# ── HYPERPARAMETERS ────────────────────────────────────────────────
EPOCHS       = 20        # Extended: single-cycle cosine needs more epochs to converge
WARMUP_EP    = 2         # Linear warmup before cosine decay kicks in
BASE_LR      = 3e-5      # Peak LR — single cosine cycle decays monotonically to MIN_LR
MIN_LR       = 1e-7      # Deep floor gives cosine full dynamic range
ACCUM_STEPS  = 3
EMA_DECAY    = 0.9995    # Very slow EMA → stable pseudo-labels
WEIGHT_DECAY = 0.05      # ✅ FIX: Reduced from 0.10 → 0.05; heavy L2 was preventing convergence

EARLY_STOP_PATIENCE = 6  # Slightly tighter with stable cosine (no restart noise)
epochs_no_improve   = 0

LABEL_SMOOTH_EPS = 0.05  # ✅ FIX: Increased from 0.02 → 0.05 for better generalization

# ── NOTE: SGDR REMOVED ─────────────────────────────────────────────
# Root-cause diagnosis: SGDR restarts at epochs 7, 11, 15 were kicking the
# model OUT of a valid minimum (best AUC always appeared at low-LR tail of
# each cycle). Single monotonic cosine decay is strictly better here.
# Evidence from logs: Ep9 (LR=1.5e-5) → 0.7992 BEST, then Ep11 restart
# (LR=3e-5) → model never recovers past 0.7992 for 7 more epochs.

# ── Progressive Unfreezing Schedule (ULMFiT-style) ────────────────
# ✅ FIX: Early blocks (conv1, resblocks.0-3, positional_embedding)
# are PERMANENTLY FROZEN — they encode BioMedCLIP's medical visual
# vocabulary. Unfreezing them causes catastrophic forgetting and
# hurts generalization on NIH-14.
UNFREEZE_SCHEDULE = {
    0: ["resblocks.10", "resblocks.11", "ln_post", "proj"],  # Ep 0-1: top-2 blocks
    2: ["resblocks.8",  "resblocks.9"],                       # Ep 2-3: + blocks 8-9
    4: ["resblocks.6",  "resblocks.7"],                       # Ep 4-5: + blocks 6-7
    6: ["resblocks.4",  "resblocks.5"],                       # Ep 6-7: + blocks 4-5
    # ✅ resblocks.0-3, conv1, positional_embedding, class_embedding, ln_pre
    # deliberately NOT in schedule → permanently frozen to preserve BioMedCLIP features
}

RESUME_EPOCH    = 0      # set >0 to resume from checkpoint
RESUME_BEST_AUC = 0.0    # override e.g. 0.8086 to protect saved weights

NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print(f"🚀 CARDIOAI V6 TEACHER  ▸  BiomedCLIP (ADVANCED v2)  ▸  NIH-14")
print(f"   HARDWARE : {DEVICE}  |  GPUs: {num_gpus}")
print(f"   IMG SIZE : {IMG_SIZE}   BATCH: {BATCH_SIZE}")
print(f"   EPOCHS   : {EPOCHS}  (warm-up: {WARMUP_EP}, cosine decay)")
print(f"   STRATEGY : Progressive Unfreeze + Single-Cycle Cosine + 5-View TTA + RandAugment")
print("=" * 65)


# ════════════════════════════════════════════════════════════════
# 2. DATASET
# ════════════════════════════════════════════════════════════════
# ❌ FIX 1: CLAHE removed to prevent CLIP representation distortion.

def smooth_labels(hard: np.ndarray, eps: float = LABEL_SMOOTH_EPS) -> np.ndarray:
    return hard * (1.0 - eps) + (1.0 - hard) * eps

# ── 2c. Read metadata ─────────────────────────────────────────────
metadata    = pd.read_csv(META_PATH)
image_paths = glob.glob(DATA_DIR + "images_*/images/*.png")
path_dict   = {os.path.basename(p): p for p in image_paths}
metadata    = metadata[metadata["Image Index"].isin(path_dict)].reset_index(drop=True)
metadata["image_path"] = metadata["Image Index"].map(path_dict)

# 🔼 FIX 4: Label Cleaning
# Remove extremely noisy images (> 3 diseases)
metadata = metadata[metadata["Finding Labels"].apply(lambda x: len(x.split("|")) <= 3)]
# (Reverted No Finding downsample: Asymmetric Loss requires natural distribution to calibrate)

def get_hard_labels(label_str):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, d in enumerate(DISEASES):
        if d in label_str.split("|"):
            vec[i] = 1.0
    return vec

metadata["hard_labels"] = list(
    np.stack(metadata["Finding Labels"].apply(get_hard_labels).values)
)

# 🚀 ADVANCED: Feature Extraction (Age, Gender, AP/PA View)
def extract_age(age_str):
    try:
        if isinstance(age_str, str):
            age = int(''.join(filter(str.isdigit, age_str)))
            return min(age, 100) / 100.0  # Normalize 0-1
        return float(age_str) / 100.0
    except:
        return 0.5  # Default middle age

metadata["age_norm"] = metadata["Patient Age"].apply(extract_age)
metadata["is_male"]  = (metadata["Patient Gender"] == "M").astype(np.float32)
metadata["is_ap"]    = (metadata["View Position"] == "AP").astype(np.float32)

tabular_features = metadata[["age_norm", "is_male", "is_ap"]].values.astype(np.float32)
metadata["tabular_feats"] = list(tabular_features)

# Patient-level split (no data leakage)
train_pat, temp_pat = train_test_split(
    metadata["Patient ID"].unique(), test_size=0.20, random_state=42
)
val_pat, _ = train_test_split(temp_pat, test_size=0.50, random_state=42)

train_df = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)
val_df   = metadata[metadata["Patient ID"].isin(val_pat)].reset_index(drop=True)

# ── 2d. Cache builder ─────────────────────────────────────────────
def preprocess_to_cache(df: pd.DataFrame, img_size: int, cache_dir: str):
    from concurrent.futures import ThreadPoolExecutor
    if USE_PRECOMPUTED_CACHE:
        print(f"✅ Using pre-computed Kaggle Dataset at {cache_dir}. Skipping caching.")
        return
    os.makedirs(cache_dir, exist_ok=True)
    all_paths = df["image_path"].tolist()
    done = sum(1 for p in all_paths
               if os.path.exists(os.path.join(cache_dir, os.path.basename(p).replace(".png", ".npy"))))
    if done == len(all_paths):
        print(f"✅ Cache already complete ({done} images). Skipping preprocessing.")
        return
    print(f"\n🔄 Building image cache: CLAHE-REMOVED ({img_size}×{img_size}) → {cache_dir}")
    def process_one(src_path: str):
        dst_path = os.path.join(cache_dir, os.path.basename(src_path).replace(".png", ".npy"))
        if os.path.exists(dst_path): return
        gray = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            # ❌ FIX 1: Removed `gray = _CLAHE.apply(gray)`
        np.save(dst_path, gray)
    n_threads = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(tqdm(ex.map(process_one, all_paths), total=len(all_paths), desc="  💾 Caching"))
    print(f"✅ Cache complete.")

all_df = pd.concat([train_df, val_df], ignore_index=True)
preprocess_to_cache(all_df, IMG_SIZE, CACHE_DIR)
del all_df

# ── 2e. PyTorch Dataset ───────────────────────────────────────────
class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False,
                 img_size: int = IMG_SIZE, cache_dir: str = CACHE_DIR):
        self.augment  = augment
        self.img_size = img_size
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
        raw         = np.stack(df["hard_labels"].values).astype(np.float32)
        eps         = 0.0   # label smoothing disabled (ASL + pos_weights handle imbalance)
        self.labels = raw * (1.0 - eps) + (1.0 - raw) * eps
        
        self.tabular = np.stack(df["tabular_feats"].values).astype(np.float32)

        # BioMedCLIP / OpenCLIP normalisation constants
        self._mean = torch.tensor([0.48145466, 0.4578275,  0.40821073], dtype=torch.float32).view(3,1,1)
        self._std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3,1,1)

        self.aug = T.Compose([
            # ✅ FIX: Wider crop scale (0.75→1.0) forces ViT to learn globally
            T.RandomResizedCrop(self.img_size, scale=(0.75, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomAffine(degrees=7, translate=(0.03, 0.03)),
            T.ColorJitter(brightness=0.25, contrast=0.25),
            T.RandomHorizontalFlip(p=0.5),
            # ✅ FIX: RandAugment (ViT-proven policy) — stronger regularization
            # num_ops=2, magnitude=7 is conservative enough for medical imaging
            T.RandAugment(num_ops=2, magnitude=7),
            # Keep RandomErasing very light (small lesions must not be erased)
            T.RandomErasing(p=0.05, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.cache_dir, self.fnames[idx])
        gray = np.load(path)
        if gray.shape[0] != self.img_size:
            gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        img.sub_(self._mean).div_(self._std)
        if self.augment:
            img = self.aug(img)
            
        tab = torch.tensor(self.tabular[idx], dtype=torch.float32)
        return img, tab, torch.tensor(self.labels[idx], dtype=torch.float32)


NIH_POS_FREQ = torch.tensor([
    0.103, 0.025, 0.118, 0.177,  0.051, 0.056, 0.013, 0.047,
    0.041, 0.021, 0.023, 0.015,  0.030, 0.002
], dtype=torch.float32)

train_loader = DataLoader(
    ChestXrayDataset(train_df, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,           # 🔥 REVERTED: WeightedSampler removed to prevent ASL conflict
    drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True
)
val_loader = DataLoader(
    ChestXrayDataset(val_df, augment=False),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True
)
print(f"✅ DataLoaders ready | train: {len(train_loader)} batches | val: {len(val_loader)} batches")


# ════════════════════════════════════════════════════════════════
# 3. LOSS — AsymmetricLoss + Per-Class Pos-Weights
# ════════════════════════════════════════════════════════════════
NIH_POS_WEIGHTS = ((1.0 - NIH_POS_FREQ) / NIH_POS_FREQ).clamp(max=20.0).to(DEVICE)

class AsymmetricLoss(nn.Module):
    """ASL: Ben-Baruch et al., ICCV 2021."""
    def __init__(self, gamma_neg: float = 2.0, gamma_pos: float = 0.0,
                 clip: float = 0.03, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma_neg  = gamma_neg
        self.gamma_pos  = gamma_pos
        self.clip       = clip
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob   = torch.sigmoid(logits)
        prob_m = torch.clamp(prob - self.clip, min=0.0, max=1.0)
        xs_pos = targets       * torch.log(prob   + 1e-8)
        xs_neg = (1 - targets) * torch.log(1 - prob_m + 1e-8)
        pt_pos = prob
        pt_neg = 1.0 - prob_m
        loss   = -(xs_pos * (1 - pt_pos) ** self.gamma_pos
                 + xs_neg * (1 - pt_neg) ** self.gamma_neg)
        if self.pos_weight is not None:
            loss = loss * (targets * self.pos_weight + (1 - targets))
        return loss.mean()

criterion = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.03,
                           pos_weight=NIH_POS_WEIGHTS)


# ════════════════════════════════════════════════════════════════
# 4. MODEL — BioMedCLIP with Cosine Classification Head
# ════════════════════════════════════════════════════════════════
class TeacherModel(nn.Module):
    """
    BiomedCLIP ViT-B/16 + Native Cosine Classifier.
    Starts fully frozen — progressive unfreezing is handled externally
    by apply_unfreeze_schedule() before each epoch.
    """
    def __init__(self):
        super().__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        # ALL ViT layers start FROZEN — unfreezing is progressive (back-to-front)
        for param in self.clip_model.parameters():
            param.requires_grad = False


        # Unfreeze head by default
        self.clip_model.visual.proj = None  # We use our own head
        
        self.dropout = nn.Dropout(0.3)   # 🔼 Reduced from 0.4 for slightly more signal flow

        # 🚀 ADVANCED: Tabular Feature Projection (Age, Gender, View Pos)
        self.tabular_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 512)
        )
        # 🔼 FIX 7: Gated Fusion layer
        self.gate_layer = nn.Linear(512, 512)
        
        # Initialize tabular projection to output near-zero so it doesn't destabilize early training
        nn.init.constant_(self.tabular_proj[-1].weight, 0)
        nn.init.constant_(self.tabular_proj[-1].bias, 0)
        
        # 🔼 FIX 3: Stable Tabular Gate Initialization
        nn.init.zeros_(self.gate_layer.weight)
        nn.init.ones_(self.gate_layer.bias)  # sigmoid(1) ≈ 0.73 -> permits visual signal immediately

        # Cosine classifier head (always trainable from epoch 0)
        self.head_weight = nn.Parameter(torch.randn(NUM_CLASSES, 512))
        nn.init.normal_(self.head_weight, std=0.01)
        self.logit_scale = nn.Parameter(torch.tensor(14.3))   # 🔼 FIX: CLIP default 14.3 for faster init

    def forward(self, x, tab):
        image_features = self.clip_model.encode_image(x)
        
        # 🔼 FIX 7: Gated fusion
        tab_features   = self.tabular_proj(tab)
        gate           = torch.sigmoid(self.gate_layer(tab_features))
        fused_features = image_features * gate + tab_features
        fused_features = self.dropout(fused_features)
        
        x_norm = F.normalize(fused_features, dim=-1)
        w_norm = F.normalize(self.head_weight, dim=-1)
        # Clamp logit_scale [1, 20] — prevents NaN from gradient explosion
        scale  = self.logit_scale.clamp(1.0, 20.0)
        return scale * F.linear(x_norm, w_norm)

model = TeacherModel().to(DEVICE)
print(f"✅ BiomedCLIP loaded | total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

ema_model = copy.deepcopy(model)
ema_model.eval()
for param in ema_model.parameters():
    param.requires_grad = False
print(f"✅ EMA Model (decay {EMA_DECAY}) initialized")


# ════════════════════════════════════════════════════════════════
# 5. TRAINING STRATEGY — Progressive Unfreezing + SGDR + Dynamic Optimizer
# ════════════════════════════════════════════════════════════════

def apply_unfreeze_schedule(model, epoch):
    """
    Unfreezes ViT layers back-to-front per UNFREEZE_SCHEDULE.
    Returns True if any new params were unfrozen this epoch.
    """
    newly_unfrozen = False
    for milestone, keywords in UNFREEZE_SCHEDULE.items():
        if epoch >= milestone:
            for name, param in model.clip_model.visual.named_parameters():
                if not param.requires_grad:
                    if any(kw in name for kw in keywords):
                        param.requires_grad = True
                        newly_unfrozen = True
    return newly_unfrozen


def get_grouped_params(model):
    """Groups trainable ViT params into early / mid / late LR buckets."""
    early_params, mid_params, late_params = [], [], []
    for name, param in model.clip_model.visual.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["resblocks.0", "resblocks.1", "resblocks.2",
                                    "resblocks.3", "conv1", "positional_embedding",
                                    "class_embedding", "ln_pre"]):
            early_params.append(param)
        elif any(k in name for k in ["resblocks.4", "resblocks.5",
                                      "resblocks.6", "resblocks.7"]):
            mid_params.append(param)
        else:   # resblocks 8-11, ln_post, proj
            late_params.append(param)
    return early_params, mid_params, late_params


def build_optimizer(model):
    """
    Rebuilds AdamW with layer-wise LR multipliers whenever new ViT layers
    are unfrozen. Called at epoch 0 and at each unfreeze milestone.
    """
    early_p, mid_p, late_p = get_grouped_params(model)
    param_groups = []
    if early_p:
        param_groups.append({"params": early_p, "lr_mult": 0.05})  # Deep: very slow
    if mid_p:
        param_groups.append({"params": mid_p,   "lr_mult": 0.20})  # Mid: slower
    if late_p:
        param_groups.append({"params": late_p,  "lr_mult": 1.00})  # Last blocks: full
    param_groups.append({
        "params": [model.head_weight, model.logit_scale] + list(model.tabular_proj.parameters()) + list(model.gate_layer.parameters()),
        "lr_mult": 3.0   # Head & tabular branch: 3× 
    })
    return optim.AdamW(param_groups, lr=MIN_LR, weight_decay=WEIGHT_DECAY)


def get_lr_cosine(epoch: int) -> float:
    """
    ✅ FIXED: Single-cycle cosine decay (no warm restarts).
    Linear warmup for WARMUP_EP epochs, then monotonic cosine decay to MIN_LR.
    This is PROVEN superior to SGDR when the model is already in a good basin
    (SGDR restarts were kicking the model out of its best minimum every 4 epochs).
    """
    if epoch < WARMUP_EP:
        return MIN_LR + (BASE_LR - MIN_LR) * (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, base_lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * pg.get("lr_mult", 1.0)


# Apply epoch-0 unfreeze rule + build initial optimizer
apply_unfreeze_schedule(model, 0)
optimizer = build_optimizer(model)
scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))


# ════════════════════════════════════════════════════════════════
# 6. AUTO-RESUME
# ════════════════════════════════════════════════════════════════
START_EPOCH = RESUME_EPOCH
best_auc    = RESUME_BEST_AUC

if START_EPOCH > 0:
    custom_resume = f"/kaggle/input/my-checkpoints/biomed_ckpt_ep{START_EPOCH:02d}.pth"
    resume_path   = custom_resume if os.path.exists(custom_resume) else f"{CKPT_DIR}last.pth"
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
        else:
            ema_model.load_state_dict(ckpt["model"])
        # Re-apply all unfreeze milestones up to START_EPOCH before rebuilding optimizer
        for ep in range(START_EPOCH + 1):
            apply_unfreeze_schedule(model, ep)
        optimizer = build_optimizer(model)
        optimizer.load_state_dict(ckpt["optimizer"])
        best_auc = ckpt.get("best_auc", RESUME_BEST_AUC) if RESUME_BEST_AUC == 0.0 else RESUME_BEST_AUC
        print(f"✅ Resumed from Epoch {START_EPOCH}  (best AUC so far: {best_auc:.4f})")
    else:
        print(f"⚠️  Checkpoint {resume_path} not found – starting fresh.")


# ════════════════════════════════════════════════════════════════
# 7. MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════
for epoch in range(START_EPOCH, EPOCHS):

    # ── 7a. Progressive Unfreeze + Optimizer Rebuild ──────────────
    if epoch > START_EPOCH:   # epoch 0 already applied above
        newly_unfrozen = apply_unfreeze_schedule(model, epoch)
        if newly_unfrozen:
            optimizer = build_optimizer(model)
            unlocked = sum(p.requires_grad for p in model.clip_model.visual.parameters())
            print(f"  🔓 Ep{epoch}: {unlocked} ViT visual params now trainable → optimizer rebuilt")

    # ── 7b. Cosine Learning Rate (single cycle, no restarts) ─────────
    current_lr = get_lr_cosine(epoch)
    set_lr(optimizer, current_lr)
    head_lr    = current_lr * 3.0
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nEpoch {epoch+1}/{EPOCHS}  |  LR = {current_lr:.2e}  |  Head LR = {head_lr:.2e}  |  Trainable: {trainable/1e6:.1f}M")

    # ── 7c. Training ──────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc="  [Train]", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for i, (images, tab, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=True)
        tab    = tab.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # MixUp 20% of batches (matches ConvNeXt teacher proven rate)
        # NIH-14 is multi-label/noisy; 50% MixUp destabilizes the cosine head.
        use_mixup = (epoch >= WARMUP_EP) and (np.random.random() > 0.8)
        if use_mixup:
            lam = float(np.random.beta(0.4, 0.4))
            idx = torch.randperm(images.size(0), device=DEVICE)
            images = lam * images + (1 - lam) * images[idx]
            # ❌ FIX 2: DO NOT mix tabular features, it adds meaningless noise!
            labels = lam * labels + (1 - lam) * labels[idx]

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images, tab)
            loss   = criterion(logits, labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if ((i + 1) % ACCUM_STEPS == 0) or ((i + 1) == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA update (decay=0.9995 → very slow tracking → stable)
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(model_p.data, alpha=1.0 - EMA_DECAY)

        epoch_loss += loss.item() * ACCUM_STEPS
        pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}", lr=f"{current_lr:.1e}")

    print(f"  Train loss: {epoch_loss / len(train_loader):.4f}")

    # ── 7d. Validation — EMA + 5-View TTA ────────────────────────
    ema_model.eval()
    val_preds, val_targs = [], []

    with torch.no_grad():
        for images, tab, labels in tqdm(val_loader, desc="  [Val]  ", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            tab    = tab.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            h, w   = images.shape[2], images.shape[3]

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                # 5-View TTA: original | h-flip | centre-90% | top-left-90% | bot-right-90%
                p1 = torch.sigmoid(ema_model(images, tab))
                p2 = torch.sigmoid(ema_model(torch.flip(images, dims=[3]), tab))
                ch, cw   = int(h * 0.9), int(w * 0.9)
                sh_, sw_ = (h - ch) // 2, (w - cw) // 2
                crop_c   = F.interpolate(images[:, :, sh_:sh_+ch, sw_:sw_+cw],
                                         size=(h, w), mode="bilinear", align_corners=False)
                p3 = torch.sigmoid(ema_model(crop_c, tab))
                crop_tl  = F.interpolate(images[:, :, :ch, :cw],
                                         size=(h, w), mode="bilinear", align_corners=False)
                p4 = torch.sigmoid(ema_model(crop_tl, tab))
                crop_br  = F.interpolate(images[:, :, h-ch:, w-cw:],
                                         size=(h, w), mode="bilinear", align_corners=False)
                p5 = torch.sigmoid(ema_model(crop_br, tab))
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

    # ── 7e. Checkpoint / Early Stopping ──────────────────────────
    if macro_auc > best_auc:
        best_auc = macro_auc
        epochs_no_improve = 0
        core_state = (ema_model.module.state_dict()
                      if hasattr(ema_model, "module") else ema_model.state_dict())
        torch.save(core_state, f"{CKPT_DIR}best_biomedclip_teacher.pth")
        print(f"  🏆 NEW BEST TEACHER  AUC: {best_auc:.4f}  → best_biomedclip_teacher.pth")
    else:
        epochs_no_improve += 1
        print(f"  ⚠️ No improvement for {epochs_no_improve} epoch(s).")

    # Safety checkpoint (overwrite last.pth to save disk space)
    torch.save({
        "epoch"    : epoch + 1,
        "model"    : model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_auc" : best_auc,
        "auc"      : macro_auc,
    }, f"{CKPT_DIR}last.pth")

    # ── 7f. CPU/GPU Memory Cleanup ───────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  🧹 CPU/GPU cache cleared (Epoch {epoch+1} done)")

    # ── 7g. Thread-safe Cooldown ──────────────────────────────────
    time.sleep(5)

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n🛑 EARLY STOPPING: No improvement for {EARLY_STOP_PATIENCE} consecutive epochs.")
        break

print("=" * 65)

# ── 8. Threshold Tuning (F1 Optimization) ────────────────────────
from sklearn.metrics import f1_score
print("\n🔍 Tuning Optimal Thresholds per Class...")
best_thresh = []
for i in range(NUM_CLASSES):
    best, best_t = 0, 0.5
    for t in np.linspace(0.01, 0.99, 100):
        try:
            score = f1_score(val_targs[:, i], val_preds[:, i] > t, zero_division=0)
            if score > best:
                best, best_t = score, t
        except ValueError:
            pass
    best_thresh.append(best_t)
print("🎯 Best Thresholds per Class:", {k: round(v, 2) for k, v in zip(DISEASES, best_thresh)})

if os.path.exists(CACHE_DIR) and "working" in CACHE_DIR:
    print(f"\n📁 Leaving .npy cache in {CACHE_DIR} for reuse.")


# ════════════════════════════════════════════════════════════════
# 9. PSEUDO-LABELING FOR KNOWLEDGE DISTILLATION
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("🎯 STARTING PSEUDO-LABELING FOR DISTILLATION")
print("   Generating soft-labels for the entire training set (100%)...")
print("=" * 65)

train_df_full = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)

class PseudoLabelDataset(Dataset):
    """Fast pseudo-label inference using the .npy cache."""
    def __init__(self, df: pd.DataFrame, img_size: int = IMG_SIZE):
        self.img_paths = df["image_path"].tolist()
        self.img_size  = img_size
        self._mean     = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self._std      = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        self.tabular   = np.stack(df["tabular_feats"].values).astype(np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # ❌ FIX 1: Removed CLAHE
        img = torch.from_numpy(gray).unsqueeze(0).repeat(3, 1, 1).float().div_(255.0)
        img.sub_(self._mean).div_(self._std)
        tab = torch.tensor(self.tabular[idx], dtype=torch.float32)
        return img, tab

pseudo_dataset = PseudoLabelDataset(train_df_full, img_size=IMG_SIZE)
pseudo_loader  = DataLoader(
    pseudo_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
    num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4
)

# Load best EMA weights for cleanest pseudo-labels
try:
    ema_model.load_state_dict(
        torch.load(f"{CKPT_DIR}best_biomedclip_teacher.pth", map_location=DEVICE, weights_only=False)
    )
    print(f"✅ Loaded best EMA weights for pseudo-labeling.")
except Exception as e:
    print(f"⚠️ Could not load best weights: {e}. Using current ema_model.")

ema_model.eval()
all_soft_labels = []
print(f"⏳ Inferring soft-labels (w/ TTA) for {len(train_df_full)} images...")
with torch.no_grad():
    for images, tab in tqdm(pseudo_loader, desc="  [Pseudo-Labeling]"):
        images = images.to(DEVICE, non_blocking=True)
        tab    = tab.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
            probs1 = torch.sigmoid(ema_model(images, tab))
            probs2 = torch.sigmoid(ema_model(torch.flip(images, dims=[3]), tab))
            probs  = (probs1 + probs2) / 2.0
        all_soft_labels.append(probs.cpu())

all_soft_labels = torch.cat(all_soft_labels).numpy()

# Export raw numpy (for student ensemble fusion)
np.save(f"{CKPT_DIR}clip_preds.npy", all_soft_labels)
print(f"📦 Saved raw probabilities → {CKPT_DIR}clip_preds.npy")

# 🔼 FIX 5: Dynamic Percentile-Based Pseudo-label Thresholding
high_thresh = np.zeros(NUM_CLASSES)
low_thresh  = np.zeros(NUM_CLASSES)

for i in range(NUM_CLASSES):
    # Dynamic thresholds based on this specific model's distribution
    high_thresh[i] = np.percentile(all_soft_labels[:, i], 85)
    low_thresh[i]  = np.percentile(all_soft_labels[:, i], 15)

mask         = (all_soft_labels > high_thresh) | (all_soft_labels < low_thresh)
hard_labels  = np.stack(train_df_full["hard_labels"].values).astype(np.float32)
final_labels = 0.7 * hard_labels + 0.3 * all_soft_labels
final_labels[~mask] = -1.0

soft_cols = [f"soft_{d}" for d in DISEASES]
soft_df   = pd.DataFrame(final_labels, columns=soft_cols)
pseudo_df = pd.concat(
    [train_df_full[["Image Index", "Patient ID", "Finding Labels"]], soft_df],
    axis=1
)
pseudo_csv_path = f"{CKPT_DIR}nih_pseudo_labels.csv"
pseudo_df.to_csv(pseudo_csv_path, index=False)

print(f"🔥 Pseudo-labeling complete! Saved to: {pseudo_csv_path}")
print("   Upload this CSV + best weights for Student Knowledge Distillation!")
print("=" * 65 + "\n")
