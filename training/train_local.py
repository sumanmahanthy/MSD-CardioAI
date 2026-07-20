"""
=============================================================================
 train_local.py — Local Training Script for CardioAI Pro
=============================================================================
 Run from the project root:
   cd f:/Mtech/Cardiomegaly_Pro
   python train_local.py

 Requirements:
   pip install torch torchvision timm opencv-python scikit-learn tqdm pandas

 Expected folder structure for your NIH dataset:
   nih_dataset/
   ├── Data_Entry_2017.csv
   └── images/            ← all .png files (can be in sub-folders)
       ├── 00000001_000.png
       ├── 00000001_001.png
       └── ...

 Output (saved in backend/models/):
   ├── ssl_vit_encoder.pth        ← after Phase 1 (SSL)
   └── vit_cardiomegaly_384.pth   ← after Phase 2 (Fine-tuning)
=============================================================================
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# ✏️  CONFIGURE THESE PATHS FOR YOUR LOCAL NIH DATASET
# ─────────────────────────────────────────────────────────────────────────────
NIH_DATASET_PATH = r"C:\path\to\your\nih_dataset"   # ← change this
CSV_PATH         = os.path.join(NIH_DATASET_PATH, "Data_Entry_2017.csv")
IMAGES_DIR       = os.path.join(NIH_DATASET_PATH, "images")   # or images_001, etc.

# Output: where to save trained weights (picked up by the FastAPI backend)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "backend", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters — safe defaults for 8 GB GPU
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Phase 1 — SSL Pre-training
    "ssl_epochs":       3,
    "ssl_batch_size":   16,   # reduce to 8 if you get CUDA OOM
    "ssl_lr":           1e-4,
    "ssl_temperature":  0.5,

    # Phase 2 — Fine-tuning
    "ft_epochs":        10,
    "ft_batch_size":    16,   # reduce to 8 if you get CUDA OOM
    "ft_lr":            1e-4,

    # Data
    "image_size":       384,
    "val_split":        0.2,
    "random_seed":      42,

    # Hardware
    "num_workers":      2,    # set to 0 on Windows if DataLoader hangs
    "pin_memory":       True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  CardioAI Pro — Local Training")
print(f"  Device : {device}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load CSV and build file index
# ─────────────────────────────────────────────────────────────────────────────
def load_nih_dataframe(csv_path: str, images_dir: str) -> pd.DataFrame:
    print("📂 Loading NIH ChestX-ray14 CSV...")
    df = pd.read_csv(csv_path)

    # Binary Cardiomegaly label (matches Kaggle code exactly)
    df["Cardiomegaly"] = df["Finding Labels"].apply(
        lambda x: 1 if "Cardiomegaly" in x else 0
    )
    df = df[["Image Index", "Cardiomegaly"]]

    # Build filename → full path index
    # Supports both flat images/ and images_001/, images_002/ structure
    print("🔍 Scanning image files (this may take a moment for 112K images)...")
    image_paths = glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True)
    image_dict  = {os.path.basename(p): p for p in image_paths}

    df["path"] = df["Image Index"].map(image_dict)
    df = df.dropna(subset=["path"]).reset_index(drop=True)

    total     = len(df)
    positive  = df["Cardiomegaly"].sum()
    print(f"✅ Dataset loaded: {total} images | "
          f"Cardiomegaly: {positive} ({100*positive/total:.1f}%) | "
          f"Normal: {total - positive} ({100*(total-positive)/total:.1f}%)\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — CLAHE Preprocessing (matches Kaggle preprocess_image exactly)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(img_path: str, size: int = 384) -> np.ndarray:
    """
    Kaggle exact replica:
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, (384, 384))
      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
      img = clahe.apply(img)
      img = img / 255.0
      img = np.expand_dims(img, axis=0)   → (1, 384, 384)
      img = np.repeat(img, 3, axis=0)     → (3, 384, 384)
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Dataset Classes
# ─────────────────────────────────────────────────────────────────────────────
class CardiomegalyDataset(Dataset):
    """
    Supervised dataset for fine-tuning.
    Matches Kaggle's ChestXrayDataset exactly.
    """
    def __init__(self, dataframe: pd.DataFrame, image_size: int = 384):
        self.df         = dataframe.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = preprocess_image(row["path"], self.image_size)
        img   = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(row["Cardiomegaly"], dtype=torch.long)
        return img, label


class SSLDataset(Dataset):
    """
    Self-supervised dataset — returns two augmented views of the same image.
    Matches Kaggle's SSLChestDataset.
    """
    import torchvision.transforms as T

    _ssl_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        image    = Image.open(img_path).convert("RGB")
        return self.T.Compose(self._ssl_transform.transforms)(image), \
               self.T.Compose(self._ssl_transform.transforms)(image)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — SimCLR Model (matches Kaggle exactly)
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder   = encoder
        self.projector = ProjectionHead()

    def forward(self, x):
        return self.projector(self.encoder(x))


def contrastive_loss(z1, z2, temperature=0.5):
    """NT-Xent loss (matches Kaggle exactly)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    similarity = torch.mm(z1, z2.T) / temperature
    labels     = torch.arange(z1.size(0)).to(device)
    return F.cross_entropy(similarity, labels)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — SSL Pre-training
# ─────────────────────────────────────────────────────────────────────────────
def phase1_ssl_pretraining(df: pd.DataFrame) -> str:
    """
    Train SimCLR on unlabeled X-rays for {ssl_epochs} epochs.
    Saves: backend/models/ssl_vit_encoder.pth
    """
    print("\n" + "="*60)
    print("  PHASE 1 — SSL Pre-Training (SimCLR)")
    print("="*60)

    ssl_dataset = SSLDataset(df)
    ssl_loader  = DataLoader(
        ssl_dataset,
        batch_size  = CONFIG["ssl_batch_size"],
        shuffle     = True,
        num_workers = CONFIG["num_workers"],
        pin_memory  = CONFIG["pin_memory"],
        drop_last   = True,   # NT-Xent needs even batches
    )

    encoder   = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model_ssl = SimCLR(encoder).to(device)
    optimizer = torch.optim.Adam(model_ssl.parameters(), lr=CONFIG["ssl_lr"])

    for epoch in range(CONFIG["ssl_epochs"]):
        model_ssl.train()
        total_loss = 0.0
        pbar = tqdm(ssl_loader, desc=f"SSL Epoch {epoch+1}/{CONFIG['ssl_epochs']}")
        for img1, img2 in pbar:
            img1, img2 = img1.to(device), img2.to(device)
            z1, z2     = model_ssl(img1), model_ssl(img2)
            loss       = contrastive_loss(z1, z2, CONFIG["ssl_temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"  Epoch {epoch+1} avg loss: {total_loss/len(ssl_loader):.4f}")

    save_path = os.path.join(OUTPUT_DIR, "ssl_vit_encoder.pth")
    torch.save(model_ssl.encoder.state_dict(), save_path)
    print(f"\n✅ SSL encoder saved → {save_path}\n")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Fine-Tuning for Cardiomegaly
# ─────────────────────────────────────────────────────────────────────────────
def phase2_finetune(df: pd.DataFrame, ssl_encoder_path: str | None = None) -> str:
    """
    Fine-tune ViT for binary Cardiomegaly classification.
    Uses class-weighted loss + WeightedRandomSampler for class imbalance.
    Saves: backend/models/vit_cardiomegaly_384.pth
    """
    print("\n" + "="*60)
    print("  PHASE 2 — Fine-Tuning for Cardiomegaly Detection")
    print("="*60)

    # ── Train / Val split (stratified) ─────────────────────────────────────
    train_df, val_df = train_test_split(
        df, test_size=CONFIG["val_split"],
        random_state=CONFIG["random_seed"],
        stratify=df["Cardiomegaly"],
    )
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = CardiomegalyDataset(train_df, CONFIG["image_size"])
    val_ds   = CardiomegalyDataset(val_df,   CONFIG["image_size"])

    # ── WeightedRandomSampler (handles class imbalance) ────────────────────
    class_counts   = train_df["Cardiomegaly"].value_counts().sort_index().values
    class_w        = 1.0 / class_counts
    sample_weights = train_df["Cardiomegaly"].map(
        {0: class_w[0], 1: class_w[1]}
    ).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["ft_batch_size"],
        sampler=sampler, num_workers=CONFIG["num_workers"], pin_memory=CONFIG["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["ft_batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"],
    )

    # ── Model: Load SSL encoder → add classification head ────────────────
    model = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=2)

    if ssl_encoder_path and os.path.exists(ssl_encoder_path):
        print(f"  Loading SSL encoder weights from {ssl_encoder_path}...")
        encoder_weights = torch.load(ssl_encoder_path, map_location=device)
        model.load_state_dict(encoder_weights, strict=False)
        print("  ✅ SSL weights transferred (strict=False)")
    else:
        print("  ⚠️  SSL encoder not found — using ImageNet pretrained weights")
        model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=2)

    model = model.to(device)

    # ── Loss: Class-weighted CrossEntropy ────────────────────────────────
    labels_arr   = train_df["Cardiomegaly"].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_arr),
        y=labels_arr,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer     = torch.optim.Adam(model.parameters(), lr=CONFIG["ft_lr"])
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["ft_epochs"]
    )

    best_auc = 0.0
    save_path = os.path.join(OUTPUT_DIR, "vit_cardiomegaly_384.pth")

    for epoch in range(CONFIG["ft_epochs"]):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{CONFIG['ft_epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validate ──────────────────────────────────────────────────────
        acc, auc = evaluate(model, val_loader)
        scheduler.step()

        print(f"  Epoch {epoch+1:2d} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Acc: {acc*100:.2f}% | "
              f"Val AUC: {auc:.4f}")

        # ── Save best model ───────────────────────────────────────────────
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            print(f"  💾 Best model saved (AUC={best_auc:.4f}) → {save_path}")

    print(f"\n✅ Fine-tuning complete. Best AUC: {best_auc:.4f}")
    print(f"   Model saved → {save_path}\n")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    """Returns (accuracy, AUC-ROC)."""
    model.eval()
    correct, total = 0, 0
    y_true, y_scores = [], []

    with torch.no_grad():
        for images, labels in loader:
            images   = images.to(device)
            outputs  = model(images)
            probs    = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds    = outputs.argmax(dim=1).cpu().numpy()
            y_scores.extend(probs)
            y_true.extend(labels.numpy())
            correct += (preds == labels.numpy()).sum()
            total   += len(labels)

    acc = correct / total
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.0
    return acc, auc


def print_confusion_matrix(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in loader:
            images   = images.to(device)
            outputs  = model(images)
            preds    = outputs.argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.numpy())
    cm = confusion_matrix(labels_all, preds_all)
    print("\n  Confusion Matrix:")
    print(f"  {'':20s} Predicted Normal  Predicted Cardio")
    print(f"  {'Actual Normal':20s} {cm[0,0]:^16d}  {cm[0,1]:^15d}")
    print(f"  {'Actual Cardio':20s} {cm[1,0]:^16d}  {cm[1,1]:^15d}")
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Sensitivity (Recall) : {tp/(tp+fn)*100:.1f}%")
    print(f"  Specificity          : {tn/(tn+fp)*100:.1f}%")
    print(f"  Precision            : {tp/(tp+fp)*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Validate dataset path ─────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"\n❌ ERROR: CSV not found at {CSV_PATH}")
        print("   Please update NIH_DATASET_PATH at the top of this file.\n")
        exit(1)

    start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    df = load_nih_dataframe(CSV_PATH, IMAGES_DIR)

    # ── Phase 1: SSL Pre-training ─────────────────────────────────────────
    ssl_path = phase1_ssl_pretraining(df)

    # ── Phase 2: Fine-Tuning ──────────────────────────────────────────────
    ft_path = phase2_finetune(df, ssl_encoder_path=ssl_path)

    elapsed = time.time() - start
    print("\n" + "="*60)
    print(f"  🎉 Training Complete in {elapsed/3600:.1f} hours")
    print(f"  SSL encoder  → {ssl_path}")
    print(f"  Fine-tuned   → {ft_path}")
    print(f"\n  ► Restart the FastAPI backend to load these weights.")
    print(f"    cd backend && uvicorn main:app --reload")
    print("="*60 + "\n")
