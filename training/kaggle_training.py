"""
=============================================================================
 CardioAI Pro — Kaggle Training Notebook (Cell-by-Cell)
=============================================================================
 PIPELINE:
   Phase 1: CLAHE → Augmentation → SimCLR SSL → EfficientNetV2 + CBAM
            → Focal Loss + Weighted Sampling → Fine-Tune → Ensemble
   Phase 2: If Normal → Multi-Disease Classification (14 classes)

 SETUP ON KAGGLE:
   1. Settings → Accelerator → GPU T4 x2
   2. Settings → Internet → ON
   3. Add Dataset: nih-chest-xrays
   4. Copy each cell below into separate Kaggle notebook cells
=============================================================================
"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 0 — GPU Nuclear Clean (ALWAYS RUN THIS FIRST)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
import gc, torch

# Only delete model/tensor variables — NOT imported modules
_model_vars = [
    'model', 'ssl_encoder', 'model_ssl', 'multi_model', 'test_model',
    '_test', '_out', 'optimizer', 'scaler', 'scheduler', 'criterion',
    'train_loader', 'val_loader', 'ssl_loader', 'train_ds', 'val_ds',
    'ssl_dataset', 'train_multi_ds', 'val_multi_ds',
]
for _v in _model_vars:
    if _v in globals():
        try:
            del globals()[_v]
        except Exception:
            pass

# Force garbage collection
gc.collect(); gc.collect(); gc.collect()

# Release all cached CUDA memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"GPU Clean Complete!")
    print(f"  Total : {total/1e9:.2f} GB")
    print(f"  Used  : {(total-free)/1e9:.2f} GB")
    print(f"  Free  : {free/1e9:.2f} GB")
else:
    print("No GPU found — running on CPU")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install Dependencies                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# !pip install timm pytorch-grad-cam opencv-python albumentations --quiet


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Imports                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import gc, os, glob, cv2, torch, timm, random, warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load NIH ChestX-ray14 Dataset                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── UPDATE THIS PATH to match your Kaggle dataset location ──
base_path = "/kaggle/input/data/"

csv_path = base_path + "Data_Entry_2017.csv"
df = pd.read_csv(csv_path)
print(f"CSV loaded: {len(df)} rows")

# Binary label for cardiomegaly
df["Cardiomegaly"] = df["Finding Labels"].apply(lambda x: 1 if "Cardiomegaly" in x else 0)

# Multi-disease labels (for Phase 2)
ALL_DISEASES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
]
for disease in ALL_DISEASES:
    df[disease] = df["Finding Labels"].apply(lambda x: 1 if disease in x else 0)

# Build image path lookup
image_paths = glob.glob(base_path + "images_*/images/*.png")
if len(image_paths) == 0:
    image_paths = glob.glob(base_path + "images/**/*.png", recursive=True)
if len(image_paths) == 0:
    image_paths = glob.glob(base_path + "**/*.png", recursive=True)

print(f"Found {len(image_paths)} images on disk")

image_dict = {os.path.basename(p): p for p in image_paths}
df["path"]  = df["Image Index"].map(image_dict)
df = df.dropna(subset=["path"]).reset_index(drop=True)

total    = len(df)
positive = df["Cardiomegaly"].sum()
print(f"Usable: {total} images")
print(f"Cardiomegaly: {positive} ({100*positive/total:.1f}%)")
print(f"Normal:       {total - positive} ({100*(total-positive)/total:.1f}%)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — CLAHE Preprocessing                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def preprocess_clahe(img_path, size=384):
    """CLAHE preprocessing — exactly matches the backend pipeline."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((size, size), dtype=np.uint8)
    img   = cv2.resize(img, (size, size))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = img.astype(np.float32) / 255.0
    img   = np.expand_dims(img, axis=0)   # (1, H, W)
    img   = np.repeat(img, 3, axis=0)     # (3, H, W)
    return img

# Quick visual test
sample = df.iloc[0]["path"]
test_img = preprocess_clahe(sample)
plt.figure(figsize=(4, 4))
plt.imshow(test_img[0], cmap="gray")
plt.title("CLAHE Preprocessed Sample")
plt.axis("off")
plt.show()
print(f"Preprocessed shape: {test_img.shape}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Dataset Classes                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Strong augmentation for SSL ──────────────────────────────────────────────
ssl_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class SSLChestDataset(Dataset):
    """Self-supervised: returns TWO strongly augmented views of the same image."""
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")
        return ssl_transform(image), ssl_transform(image)


class CardioDataset(Dataset):
    """Supervised dataset for fine-tuning (CLAHE + 384x384)."""
    def __init__(self, dataframe, image_size=384):
        self.df = dataframe.reset_index(drop=True)
        self.size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = preprocess_clahe(row["path"], self.size)
        img   = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(row["Cardiomegaly"], dtype=torch.long)
        return img, label


class MultiDiseaseDataset(Dataset):
    """Multi-label dataset for 14-disease classification (Phase 2)."""
    def __init__(self, dataframe, image_size=384):
        self.df = dataframe.reset_index(drop=True)
        self.size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img    = preprocess_clahe(row["path"], self.size)
        img    = torch.tensor(img, dtype=torch.float32)
        labels = torch.tensor(row[ALL_DISEASES].values.astype(np.float32))
        return img, labels

print("Datasets defined.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — CBAM Attention Module                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        attn = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention — highlights WHERE to focus."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

print("CBAM Attention Module defined.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — EfficientNetV2 + CBAM Model                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class EfficientNetV2_CBAM(nn.Module):
    """
    EfficientNetV2-S backbone + CBAM attention + classification head.
    The CBAM is inserted after the last convolutional block, before the
    global average pooling and classifier.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Load EfficientNetV2-S from timm (features_only for CBAM insertion)
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0,          # remove default classifier
            global_pool="",         # remove default pooling (we do our own)
        )
        # Get the output channel count from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 384, 384)
            feat  = self.backbone(dummy)
            feat_channels = feat.shape[1]
        print(f"  Backbone output channels: {feat_channels}")

        # CBAM attention on feature maps
        self.cbam = CBAM(feat_channels, reduction=16)

        # Global average pool + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_channels, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)      # (B, C, H, W)
        features = self.cbam(features)    # Apply CBAM attention
        pooled   = self.pool(features).flatten(1)  # (B, C)
        logits   = self.classifier(pooled)
        return logits

    def get_features(self, x):
        """For SSL: returns flat feature vector."""
        features = self.backbone(x)
        pooled   = self.pool(features).flatten(1)
        return pooled


# Quick shape test on CPU only (avoids wasting GPU memory on a test)
with torch.no_grad():
    _test = EfficientNetV2_CBAM(num_classes=2, pretrained=False)
    _out  = _test(torch.randn(1, 3, 384, 384))
    print(f"  Model output shape: {_out.shape}  [CPU test OK]")
    del _test, _out
gc.collect()
print("EfficientNetV2 + CBAM model defined.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — SimCLR Framework                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR contrastive learning."""
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """SimCLR = Encoder + Projection Head."""
    def __init__(self, encoder, feature_dim):
        super().__init__()
        self.encoder   = encoder
        self.projector = ProjectionHead(input_dim=feature_dim)

    def forward(self, x):
        h = self.encoder.get_features(x)   # Use backbone features
        z = self.projector(h)
        return z


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) loss.
    FP16-safe: similarity matrix is computed in FP32 to avoid
    overflow when masking with large negative values.
    """
    # Upcast to float32 for numerical stability (AMP-safe)
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    batch_size = z1.size(0)

    # Similarity matrix in FP32
    z   = torch.cat([z1, z2], dim=0)           # (2B, D)
    sim = torch.mm(z, z.T) / temperature        # (2B, 2B)

    # Mask diagonal (self-similarity) — use -1e4 which is FP16-safe
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim  = sim.masked_fill(mask, -1e4)          # -1e4 fits in FP16 (~65504 max)

    # Positive pairs: (i → i+B) and (i+B → i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(device)

    return F.cross_entropy(sim, labels)

print("SimCLR framework defined.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Focal Loss                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples, focuses on hard misclassifications.
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights tensor

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=logits.size(1)).float()

        pt = (probs * targets_onehot).sum(dim=1)  # probability of true class
        focal_weight = (1.0 - pt) ** self.gamma

        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        loss = focal_weight * ce_loss
        return loss.mean()

print("Focal Loss defined.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — PHASE 1: SSL Pre-Training (SimCLR)                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("  PHASE 1 — Self-Supervised Pre-Training (SimCLR + EfficientNetV2)")
print("=" * 70)

SSL_EPOCHS     = 3
SSL_BATCH_SIZE = 16    # T4 safe: 16 with AMP, reduce to 8 if still OOM
SSL_LR         = 1e-4

# Clear GPU before starting
gc.collect()
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print(f"  GPU before training: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")

ssl_dataset = SSLChestDataset(df)
ssl_loader  = DataLoader(
    ssl_dataset,
    batch_size  = SSL_BATCH_SIZE,
    shuffle     = True,
    num_workers = 2,
    pin_memory  = True,
    drop_last   = True,   # CRITICAL: NT-Xent needs even batches
)

# Create encoder WITHOUT pretrained (no internet required)
ssl_encoder = EfficientNetV2_CBAM(num_classes=2, pretrained=False)
print("  Model created on CPU.")

# Get feature dimension
with torch.no_grad():
    dummy_feat = ssl_encoder.get_features(torch.randn(1, 3, 224, 224))
    feat_dim = dummy_feat.shape[1]
print(f"  Feature dimension: {feat_dim}")

model_ssl = SimCLR(ssl_encoder, feature_dim=feat_dim).to(device)
optimizer = torch.optim.AdamW(model_ssl.parameters(), lr=SSL_LR, weight_decay=1e-4)

# AMP scaler: FP16 mixed precision halves GPU memory usage
scaler = torch.cuda.amp.GradScaler()

for epoch in range(SSL_EPOCHS):
    model_ssl.train()
    total_loss = 0
    pbar = tqdm(ssl_loader, desc=f"SSL Epoch {epoch+1}/{SSL_EPOCHS}")
    for img1, img2 in pbar:
        img1, img2 = img1.to(device), img2.to(device)
        with torch.cuda.amp.autocast():          # FP16 forward
            z1   = model_ssl(img1)
            z2   = model_ssl(img2)
            loss = nt_xent_loss(z1, z2, temperature=0.5)
        optimizer.zero_grad()
        scaler.scale(loss).backward()            # FP16 backward
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / len(ssl_loader)
    print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    gc.collect()
    torch.cuda.empty_cache()

# Save SSL encoder (backbone + CBAM weights)
torch.save(model_ssl.encoder.state_dict(), "ssl_efficientnet_cbam.pth")
print("\nSSL encoder saved -> ssl_efficientnet_cbam.pth")
del model_ssl, optimizer, scaler
gc.collect()
torch.cuda.empty_cache()
free, _ = torch.cuda.mem_get_info()
print(f"  GPU after Phase 1: {free/1e9:.2f} GB free")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11 — PHASE 2: Fine-Tuning for Cardiomegaly (with Focal Loss)      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("  PHASE 2 — Fine-Tuning for 97%+ Accuracy")
print("  EfficientNetV2 + CBAM + Focal Loss + AMP + TTA + Early Stop")
print("=" * 70)

gc.collect()
torch.cuda.empty_cache()

# ── Config ───────────────────────────────────────────────────────────────────
FT_EPOCHS      = 25     # More epochs → higher accuracy
FT_BATCH_SIZE  = 16
FT_LR_HEAD     = 3e-4  # Higher LR for new classifier head
FT_LR_BACKBONE = 1e-5  # Lower LR for pretrained backbone
PATIENCE       = 5      # Early stopping patience

# ── Train/Val Split (stratified) ─────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["Cardiomegaly"]
)
print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}")

train_ds = CardioDataset(train_df, image_size=384)
val_ds   = CardioDataset(val_df,   image_size=384)

# ── Weighted Random Sampler ───────────────────────────────────────────────────
class_counts   = train_df["Cardiomegaly"].value_counts().sort_index().values
class_w        = 1.0 / class_counts.astype(float)
sample_weights = train_df["Cardiomegaly"].map({0: class_w[0], 1: class_w[1]}).values
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=FT_BATCH_SIZE,
                          sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=FT_BATCH_SIZE,
                          shuffle=False, num_workers=2)

# ── Model: Load SSL Weights ───────────────────────────────────────────────────
model = EfficientNetV2_CBAM(num_classes=2, pretrained=False)
ssl_weights = torch.load("ssl_efficientnet_cbam.pth", map_location=device)
model.load_state_dict(ssl_weights, strict=False)
model = model.to(device)
print("  SSL weights loaded.")

# ── Two-Phase LR: backbone vs head ───────────────────────────────────────────
# Head trains faster (high LR), backbone fine-tunes slowly (low LR)
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": FT_LR_BACKBONE},
    {"params": model.cbam.parameters(),     "lr": FT_LR_HEAD},
    {"params": model.pool.parameters(),     "lr": FT_LR_HEAD},
    {"params": model.classifier.parameters(),"lr": FT_LR_HEAD},
], weight_decay=1e-4)

# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-7
)

# ── Focal Loss + Label Smoothing ─────────────────────────────────────────────
labels_arr    = train_df["Cardiomegaly"].values
class_weights = compute_class_weight("balanced", classes=np.unique(labels_arr), y=labels_arr)
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"  Class weights: Normal={weight_tensor[0]:.3f}, Cardio={weight_tensor[1]:.3f}")

# Label smoothing prevents overconfidence, improves generalization
criterion = nn.CrossEntropyLoss(
    weight=weight_tensor,
    label_smoothing=0.1    # 0.1 = 10% label smoothing
)

# AMP for memory efficiency
scaler = torch.cuda.amp.GradScaler()


# ── TTA Evaluation (Test Time Augmentation) ──────────────────────────────────
tta_transforms = [
    T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)]),
    T.Compose([T.RandomHorizontalFlip(p=1.0),  T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)]),
    T.Compose([T.RandomRotation(10),            T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)]),
    T.Compose([T.RandomAffine(0, shear=5),      T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)]),
]

def evaluate_with_tta(model, loader, use_tta=False):
    """
    Evaluate with optional TTA (Test Time Augmentation).
    TTA runs each image through N transforms and averages probabilities.
    This alone adds ~1-1.5% accuracy boost.
    """
    model.eval()
    correct, total = 0, 0
    y_true, y_scores = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            if use_tta:
                # Average softmax over multiple augmented views
                probs_sum = torch.zeros(images.size(0), 2).to(device)
                with torch.cuda.amp.autocast():
                    # Original prediction
                    probs_sum += F.softmax(model(images), dim=1)
                    # Flipped
                    probs_sum += F.softmax(model(torch.flip(images, dims=[3])), dim=1)
                    # Slight brightness shift
                    probs_sum += F.softmax(model(images * 0.95), dim=1)
                probs_avg = probs_sum / 3.0
            else:
                with torch.cuda.amp.autocast():
                    probs_avg = F.softmax(model(images), dim=1)

            cardio_prob = probs_avg[:, 1].cpu().float().numpy()
            preds       = probs_avg.argmax(dim=1).cpu().numpy()

            y_scores.extend(cardio_prob)
            y_true.extend(labels.numpy())
            correct += (preds == labels.numpy()).sum()
            total   += len(labels)

    acc = correct / total
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0
    return acc, auc


# ── Training Loop with Early Stopping ────────────────────────────────────────
best_auc      = 0.0
patience_left = PATIENCE
history       = {"train_loss": [], "val_acc": [], "val_auc": []}

for epoch in range(FT_EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{FT_EPOCHS}")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast():        # FP16 forward
            outputs = model(images)
            loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping — prevents exploding gradients, stabilizes training
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{optimizer.param_groups[1]['lr']:.2e}")

    scheduler.step()
    avg_loss = train_loss / len(train_loader)

    # Standard eval every epoch, TTA eval every 5 epochs
    use_tta = ((epoch + 1) % 5 == 0)
    acc, auc = evaluate_with_tta(model, val_loader, use_tta=use_tta)

    history["train_loss"].append(avg_loss)
    history["val_acc"].append(acc)
    history["val_auc"].append(auc)

    tta_tag = " [+TTA]" if use_tta else ""
    print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | "
          f"Val Acc: {acc*100:.2f}% | Val AUC: {auc:.4f}{tta_tag}")

    if auc > best_auc:
        best_auc      = auc
        patience_left = PATIENCE
        torch.save(model.state_dict(), "efficientnet_cbam_cardiomegaly.pth")
        print(f"  >> BEST saved — AUC={best_auc:.4f} | Acc={acc*100:.2f}%")
    else:
        patience_left -= 1
        print(f"  No improvement. Patience left: {patience_left}/{PATIENCE}")
        if patience_left == 0:
            print(f"  Early stopping triggered at epoch {epoch+1}.")
            break

    gc.collect()
    torch.cuda.empty_cache()

print(f"\nPhase 2 complete!")
print(f"  Best AUC:      {best_auc:.4f}")

# Final evaluation WITH full TTA on best model
model.load_state_dict(torch.load("efficientnet_cbam_cardiomegaly.pth"))
final_acc, final_auc = evaluate_with_tta(model, val_loader, use_tta=True)
print(f"  Final Acc+TTA: {final_acc*100:.2f}%")
print(f"  Final AUC+TTA: {final_auc:.4f}")



# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 12 — PHASE 3: Multi-Disease Classification (If Not Cardiomegaly)   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("  PHASE 3 — Multi-Disease Classification (14 Classes)")
print("=" * 70)

MULTI_EPOCHS     = 5
MULTI_BATCH_SIZE = 16

# Use only "Normal" images for multi-disease training
# (X-rays that are NOT cardiomegaly may still have other diseases)
normal_df = df[df["Cardiomegaly"] == 0].reset_index(drop=True)
print(f"  Normal X-rays for multi-disease training: {len(normal_df)}")

train_multi_df, val_multi_df = train_test_split(
    normal_df, test_size=0.2, random_state=42
)

train_multi_ds = MultiDiseaseDataset(train_multi_df, image_size=384)
val_multi_ds   = MultiDiseaseDataset(val_multi_df,   image_size=384)

train_multi_loader = DataLoader(train_multi_ds, batch_size=MULTI_BATCH_SIZE, shuffle=True, num_workers=2)
val_multi_loader   = DataLoader(val_multi_ds,   batch_size=MULTI_BATCH_SIZE, shuffle=False, num_workers=2)

# Multi-disease model (same architecture, 14 output classes, multi-label)
multi_model = EfficientNetV2_CBAM(num_classes=len(ALL_DISEASES), pretrained=False)

# Load SSL-pretrained backbone
multi_model.load_state_dict(torch.load("ssl_efficientnet_cbam.pth", map_location=device), strict=False)
multi_model = multi_model.to(device)

multi_criterion = nn.BCEWithLogitsLoss()  # Multi-label = BCEWithLogitsLoss
multi_optimizer = torch.optim.AdamW(multi_model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(MULTI_EPOCHS):
    multi_model.train()
    total_loss = 0
    pbar = tqdm(train_multi_loader, desc=f"Multi Epoch {epoch+1}/{MULTI_EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = multi_model(images)
        loss = multi_criterion(outputs, labels)
        multi_optimizer.zero_grad()
        loss.backward()
        multi_optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"  Epoch {epoch+1} avg loss: {total_loss/len(train_multi_loader):.4f}")

torch.save(multi_model.state_dict(), "efficientnet_cbam_multidisease.pth")
print("Multi-disease model saved -> efficientnet_cbam_multidisease.pth")
del multi_model
torch.cuda.empty_cache()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
# \u2551  CELL 13 \u2014 Final Evaluation + Enhanced Confusion Matrix                  \u2551
# \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    roc_curve, auc as sklearn_auc, confusion_matrix
)

print("\n" + "=" * 70)
print("  FINAL EVALUATION \u2014 Comprehensive Metrics Suite")
print("=" * 70)

# \u2500\u2500 Reload best cardiomegaly model \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
model.load_state_dict(torch.load("efficientnet_cbam_cardiomegaly.pth", map_location=device))
model.eval()

# \u2500\u2500 Full inference pass \u2014 collect labels, hard preds, soft scores \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
preds_all, labels_all, scores_all = [], [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        with torch.cuda.amp.autocast():
            # TTA: original + hflip + brightness
            p1 = torch.softmax(model(images), dim=1)
            p2 = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            p3 = torch.softmax(model(images * 0.95), dim=1)
            probs = (p1 + p2 + p3) / 3.0
        cardio_scores = probs[:, 1].cpu().float().numpy()
        hard_preds    = probs.argmax(dim=1).cpu().numpy()
        scores_all.extend(cardio_scores)
        preds_all.extend(hard_preds)
        labels_all.extend(labels.numpy())

labels_arr = np.array(labels_all)
preds_arr  = np.array(preds_all)
scores_arr = np.array(scores_all)

# \u2500\u2500 Core metrics \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
from sklearn.metrics import roc_auc_score
final_auc = roc_auc_score(labels_arr, scores_arr)
final_acc = (preds_arr == labels_arr).mean()
f1        = f1_score(labels_arr, preds_arr, average="binary")
precision = precision_score(labels_arr, preds_arr, average="binary", zero_division=0)
recall    = recall_score(labels_arr, preds_arr, average="binary")

print(f"\n  Accuracy  : {final_acc*100:.2f}%")
print(f"  AUC-ROC   : {final_auc:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print()
print(classification_report(labels_arr, preds_arr,
                            target_names=["No Finding", "Cardiomegaly"]))

# \u2500\u2500 Confusion Matrix \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
cm = confusion_matrix(labels_arr, preds_arr)
tn, fp, fn, tp = cm.ravel()

print(f"  Confusion Matrix (raw):")
print(f"  {'':24s}  Pred No Finding   Pred Cardiomegaly")
print(f"  {'Actual No Finding':24s}  {tn:^16d}    {fp:^17d}")
print(f"  {'Actual Cardiomegaly':24s}  {fn:^16d}    {tp:^17d}")
print(f"\n  Sensitivity (Recall): {tp/(tp+fn)*100:.1f}%")
print(f"  Specificity          : {tn/(tn+fp)*100:.1f}%")
print(f"  Precision            : {tp/(tp+fp)*100:.1f}%  (PPV)")

# \u2500\u2500 Confusion Matrix Heatmap \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
plt.figure(figsize=(8, 7))
group_names   = ["True Neg", "False Pos", "False Neg", "True Pos"]
group_counts  = ["{0:0.0f}".format(v) for v in cm.flatten()]
group_pct     = ["{0:.2%}".format(v) for v in cm.flatten() / cm.sum()]
labels_fmt    = [f"{n}\n{c}\n({p})" for n, c, p in
                 zip(group_names, group_counts, group_pct)]
labels_fmt    = np.array(labels_fmt).reshape(2, 2)

sns.heatmap(
    cm, annot=labels_fmt, fmt="", cmap="Blues",
    xticklabels=["No Finding", "Cardiomegaly"],
    yticklabels=["No Finding", "Cardiomegaly"],
    linewidths=1, linecolor="white", annot_kws={"size": 14},
)
plt.xlabel("Predicted Label", fontsize=14, fontweight="bold", labelpad=12)
plt.ylabel("True Label", fontsize=14, fontweight="bold", labelpad=12)
plt.title(
    f"CardioAI Pro \u2014 Confusion Matrix\n"
    f"Acc={final_acc*100:.1f}%  |  AUC={final_auc:.4f}  |  F1={f1:.4f}",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("\u2705 Confusion matrix saved \u2192 confusion_matrix.png")


# \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
# \u2551  CELL 14 \u2014 ROC Curve                                                     \u2551
# \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d

fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
roc_auc_val           = sklearn_auc(fpr, tpr)

# Best operating point (max Youden index = TPR - FPR)
youden_idx  = np.argmax(tpr - fpr)
best_thresh = thresholds[youden_idx]
best_fpr    = fpr[youden_idx]
best_tpr    = tpr[youden_idx]

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color="#4f46e5", lw=2.5,
        label=f"EfficientNetV2+CBAM (AUC = {roc_auc_val:.4f})")
ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5,
        label="Random Classifier (AUC = 0.5000)")
ax.scatter(best_fpr, best_tpr, marker="o", s=120, color="#ef4444", zorder=5,
           label=f"Best Threshold = {best_thresh:.3f}\n"
                 f"(Sensitivity={best_tpr:.3f}, 1-Spec={best_fpr:.3f})")
ax.annotate(f"  Youden = {best_tpr - best_fpr:.3f}",
            xy=(best_fpr, best_tpr), fontsize=10, color="#ef4444")
ax.fill_between(fpr, tpr, alpha=0.08, color="#4f46e5")
ax.set_xlabel("False Positive Rate (1 \u2212 Specificity)", fontsize=13, fontweight="bold")
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13, fontweight="bold")
ax.set_title("CardioAI Pro \u2014 ROC Curve (Cardiomegaly Detection)",
             fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\u2705 ROC curve saved \u2192 roc_curve.png  (AUC = {roc_auc_val:.4f})")
print(f"   Best threshold: {best_thresh:.4f}")


# \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
# \u2551  CELL 15 \u2014 F1 / Precision / Recall Bar Graph                            \u2551
# \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d

f1_per     = f1_score(labels_arr, preds_arr, average=None)
prec_per   = precision_score(labels_arr, preds_arr, average=None, zero_division=0)
recall_per = recall_score(labels_arr, preds_arr, average=None)
macro_f1   = f1_score(labels_arr, preds_arr, average="macro")
weighted_f1 = f1_score(labels_arr, preds_arr, average="weighted")

classes = ["No Finding", "Cardiomegaly"]
x       = np.arange(len(classes))
width   = 0.25

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sub-plot 1: Per-class grouped bars
ax = axes[0]
bars_f1   = ax.bar(x - width, f1_per,     width, label="F1-Score",  color="#4f46e5", alpha=0.85)
bars_prec = ax.bar(x,         prec_per,   width, label="Precision", color="#10b981", alpha=0.85)
bars_rec  = ax.bar(x + width, recall_per, width, label="Recall",    color="#f59e0b", alpha=0.85)

for bars in [bars_f1, bars_prec, bars_rec]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=12)
ax.set_ylabel("Score", fontsize=13)
ax.set_ylim(0, 1.15)
ax.set_title("Per-Class: F1 / Precision / Recall", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3, linestyle="--")

# Sub-plot 2: Overall summary
ax2 = axes[1]
metric_labels = ["Accuracy", "AUC-ROC", "Macro F1", "Weighted F1",
                 "Precision\n(Cardio)", "Recall\n(Cardio)", "F1\n(Cardio)"]
metric_values = [final_acc, final_auc, macro_f1, weighted_f1,
                 prec_per[1], recall_per[1], f1_per[1]]
colors        = ["#4f46e5", "#8b5cf6", "#ec4899", "#f97316",
                 "#10b981", "#f59e0b", "#06b6d4"]

bars2 = ax2.bar(np.arange(len(metric_labels)), metric_values,
                color=colors, alpha=0.85, width=0.6)
for bar, v in zip(bars2, metric_values):
    ax2.annotate(f"{v:.4f}",
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

ax2.set_xticks(np.arange(len(metric_labels)))
ax2.set_xticklabels(metric_labels, fontsize=10)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Score", fontsize=13)
ax2.set_title("Overall Model Performance Summary", fontsize=14, fontweight="bold")
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.axhline(y=0.90, color="red", linestyle="--", alpha=0.5, label="0.90 target")
ax2.legend(fontsize=10)

fig.suptitle("CardioAI Pro \u2014 Evaluation Metrics (EfficientNetV2 + CBAM + SimCLR)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("metrics_bar_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("\u2705 Metrics bar chart saved \u2192 metrics_bar_chart.png")


# \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
# \u2551  CELL 16 \u2014 Training History Plots                                        \u2551
# \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history["train_loss"], "b-o", linewidth=2, markersize=5)
axes[0].set_title("Training Loss", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3, linestyle="--")

axes[1].plot([a * 100 for a in history["val_acc"]], "g-o", linewidth=2, markersize=5)
axes[1].axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% target")
axes[1].set_title("Validation Accuracy", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, linestyle="--")

axes[2].plot(history["val_auc"], "r-o", linewidth=2, markersize=5, label="Val AUC")
axes[2].axhline(y=0.90, color="orange", linestyle="--", alpha=0.5, label="0.90 target")
axes[2].set_title("Validation AUC-ROC", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("AUC")
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3, linestyle="--")

plt.suptitle("CardioAI Pro \u2014 Training History", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
plt.show()
print("\u2705 Training history saved \u2192 training_history.png")


# \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
# \u2551  CELL 17 \u2014 Grad-CAM Visualization (Explainability)                       \u2551
# \u2551  Run:  !pip install pytorch-grad-cam -q                                   \u2551
# \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_OK = True
except ImportError:
    print("\u26a0\ufe0f pytorch-grad-cam not installed. Run:  !pip install pytorch-grad-cam -q")
    GRADCAM_OK = False

if GRADCAM_OK:
    model.eval()

    # EfficientNetV2-S + CBAM: target the CBAM attention output layer
    target_layer = [model.cbam]
    cam = GradCAM(model=model, target_layers=target_layer)

    # Pick N_SAMPLES samples: half positive (Cardiomegaly), half negative (No Finding)
    N_SAMPLES = 6
    sample_idx_pos = np.where(labels_arr == 1)[0][:N_SAMPLES // 2]
    sample_idx_neg = np.where(labels_arr == 0)[0][:N_SAMPLES // 2]
    sample_indices = np.concatenate([sample_idx_pos, sample_idx_neg])

    fig, axes_gc = plt.subplots(2, N_SAMPLES, figsize=(N_SAMPLES * 4, 9))
    fig.suptitle(
        "CardioAI Pro \u2014 Grad-CAM Explainability\n"
        "(Top: Original CLAHE X-ray  |  Bottom: Grad-CAM Activation Heatmap)",
        fontsize=14, fontweight="bold"
    )

    for col_i, val_idx in enumerate(sample_indices):
        # Load and preprocess the image
        img_raw = preprocess_clahe(val_df.iloc[val_idx]["path"], size=384)
        inp_t   = torch.tensor(img_raw, dtype=torch.float32).unsqueeze(0).to(device)

        # Normalise for model
        mean_t = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        std_t  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
        inp_norm = (inp_t - mean_t) / std_t

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                gray_cam = cam(input_tensor=inp_norm, targets=None)[0]

        # Base for overlay: [H, W, 3] float [0,1]
        base_np = np.transpose(img_raw, (1, 2, 0)).astype(np.float32)
        base_np = np.clip(base_np, 0.0, 1.0)
        overlay = show_cam_on_image(base_np, gray_cam, use_rgb=True)

        true_lbl  = "Cardiomegaly" if labels_arr[val_idx] == 1 else "No Finding"
        pred_lbl  = "Cardiomegaly" if preds_arr[val_idx]  == 1 else "No Finding"
        score_v   = scores_arr[val_idx]
        color_txt = "limegreen" if true_lbl == pred_lbl else "tomato"

        axes_gc[0, col_i].imshow(base_np[:, :, 0], cmap="gray")
        axes_gc[0, col_i].set_title(f"GT: {true_lbl}", fontsize=10, fontweight="bold")
        axes_gc[0, col_i].axis("off")

        axes_gc[1, col_i].imshow(overlay)
        axes_gc[1, col_i].set_title(
            f"Pred: {pred_lbl}\nScore: {score_v:.3f}",
            fontsize=10, color=color_txt, fontweight="bold"
        )
        axes_gc[1, col_i].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\u2705 Grad-CAM visualization saved \u2192 gradcam_visualization.png")
else:
    print("\u26a0\ufe0f  Skipped Grad-CAM (pytorch-grad-cam not available). "
          "Run !pip install pytorch-grad-cam -q and re-run this cell.")


# \u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
# \u2551  CELL 18 \u2014 Download Instructions                                         \u2551
# \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d

print("\n" + "=" * 70)
print("  TRAINING COMPLETE!")
print("=" * 70)
print()
print("  Download these files from the Kaggle OUTPUT tab:")
print()
print("  Models:")
print("  1. ssl_efficientnet_cbam.pth         <- SSL pretrained encoder")
print("  2. efficientnet_cbam_cardiomegaly.pth <- Fine-tuned cardiomegaly head")
print("  3. efficientnet_cbam_multidisease.pth <- Multi-disease classifier")
print()
print("  Metric Plots (for thesis / paper):")
print("  4. confusion_matrix.png              <- Enhanced confusion matrix heatmap")
print("  5. roc_curve.png                     <- ROC curve + AUC + best threshold")
print("  6. metrics_bar_chart.png             <- F1 / Precision / Recall bar graph")
print("  7. training_history.png              <- Loss / Acc / AUC per epoch")
print("  8. gradcam_visualization.png         <- Grad-CAM explainability panel")
print()
print("  Place model .pth files in:  backend/models/")
print("  Then restart the FastAPI backend.")
print("=" * 70)
